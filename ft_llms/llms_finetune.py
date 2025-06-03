from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, DataCollatorForLanguageModeling
from datasets import Dataset, load_from_disk
import torch
import logging
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config
import pandas as pd
import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(here)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)
from data.prepare import dataset_prepare
from attack.utils import create_folder
from transformers import LlamaTokenizer, get_scheduler
from ft_llms.utils import constantlengthdatasetiter, print_trainable_parameters

logger = logging.getLogger("finetune")

from src.parser import parse_args

# Try to import FSDP components
try:
    from lightning.pytorch.strategies import FSDPStrategy
    import torch.distributed.fsdp
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available in this PyTorch version")


class PackedDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that packs multiple text examples into fixed-length sequences
    """
    def __init__(self, dataset, tokenizer, block_size, dataset_text_field="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dataset_text_field = dataset_text_field
        
        # Get dataset length without loading all data
        self.dataset_length = len(dataset)
        
        # Pre-compute sequence boundaries without loading all text
        logger.info(f"Computing sequence boundaries for {self.dataset_length} examples...")
        self.sequence_boundaries = []
        current_length = 0
        
        # Add BOS token length if tokenizer has it
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            current_length += 1
            
        
        for i, text in enumerate(dataset[self.dataset_text_field]):
            if not text.strip():
                continue
                
            # Estimate token length without actually tokenizing
            estimated_tokens = len(text) // 4  # rough estimate: 4 chars per token
            
            # If this would exceed block_size, mark a boundary
            if current_length + estimated_tokens + 1 >= self.block_size:  # +1 for EOS
                if current_length >= self.block_size:
                    self.sequence_boundaries.append(i)
                current_length = estimated_tokens + 1
            else:
                current_length += estimated_tokens + 1
            
            if (i + 1) % 10000 == 0:
                logger.debug(f"Processed {i + 1}/{self.dataset_length} examples")
        
        # Add final boundary
        if current_length > 0:
            self.sequence_boundaries.append(self.dataset_length)
        
        logger.info(f"Created approximately {len(self.sequence_boundaries)} packed sequences")
    
    def __len__(self):
        return len(self.sequence_boundaries)
    
    def __getitem__(self, idx):
        # Determine which examples to include in this sequence
        start_idx = self.sequence_boundaries[idx - 1] if idx > 0 else 0
        end_idx = self.sequence_boundaries[idx]
        
        # Collect and tokenize texts on-demand
        all_tokens = []
        
        # Add BOS token at the beginning
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            all_tokens.append(self.tokenizer.bos_token_id)
        
        for i in range(start_idx, min(end_idx, self.dataset_length)):
            text = self.dataset[self.dataset_text_field][i]
            if not text.strip():
                continue
                
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(self.tokenizer.eos_token_id)
        
        # Remove last EOS and add final EOS
        if all_tokens and all_tokens[-1] == self.tokenizer.eos_token_id:
            all_tokens.pop()
        if all_tokens:
            all_tokens.append(self.tokenizer.eos_token_id)
        
        # Truncate or pad to exact block_size
        if len(all_tokens) > self.block_size:
            all_tokens = all_tokens[:self.block_size]
        elif len(all_tokens) < self.block_size:
            # Pad with pad token
            pad_length = self.block_size - len(all_tokens)
            all_tokens.extend([self.tokenizer.pad_token_id] * pad_length)
        
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': torch.ones_like(input_ids)
        }
        
class LLMFineTuneModule(L.LightningModule):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        
    def forward(self, batch):
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.args.use_int4 or self.args.use_int8:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(
                self.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                eps=1e-6
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                eps=1e-6
            )
        
        if self.args.lr_scheduler_type == "linear":
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
        return optimizer


class LLMDataModule(L.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset, val_dataset = dataset_prepare(self.args, tokenizer=self.tokenizer)
            
            if self.args.refer:
                train_dataset = None
                refer_data_path = self.args.generated_dataset_dir
                train_dataset = load_from_disk(refer_data_path)
                
            if self.args.debug:
                self.args.dataset.eval.end_idx = 30
                self.args.dataset.train.end_idx = 30
                logger.warning("Debug mode is enabled, only using 30 samples for training and evaluation.")
                
            # Apply dataset slicing
            train_data = train_dataset[self.args.dataset.train.start_idx:self.args.dataset.train.end_idx]
            val_data = val_dataset[self.args.dataset.eval.start_idx:self.args.dataset.eval.end_idx]
            
            # Create packed datasets that mimic SFTTrainer behavior
            self.train_dataset = PackedDataset(
                train_data, 
                self.tokenizer, 
                self.args.block_size,
                dataset_text_field="text"
            )
            self.val_dataset = PackedDataset(
                val_data, 
                self.tokenizer, 
                self.args.block_size,
                dataset_text_field="text"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )


def get_training_strategy(args):
    """Determine the training strategy based on distributed_training config"""
    
    # Check if distributed training is enabled
    if not hasattr(args, 'distributed_training') or not args.distributed_training.use_distributed:
        return "auto"
    
    dist_config = args.distributed_training
    strategy_name = dist_config.strategy.lower()
    
    if strategy_name == "ddp":
        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    
    elif strategy_name == "fsdp":
        if not FSDP_AVAILABLE:
            logger.error("FSDP requested but not available. Falling back to DDP.")
            return DDPStrategy()
        
        # Get FSDP configuration
        fsdp_config = dist_config.fsdp_config
        
        # Determine auto wrap policy
        if fsdp_config.auto_wrap_policy == "transformer":
            auto_wrap_policy = transformer_auto_wrap_policy
        elif fsdp_config.auto_wrap_policy == "size":
            auto_wrap_policy = size_based_auto_wrap_policy
        else:
            auto_wrap_policy = None
        
        # Determine sharding strategy
        sharding_strategy_map = {
            "FULL_SHARD": torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": torch.distributed.fsdp.ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_strategy_map.get(
            fsdp_config.sharding_strategy, 
            torch.distributed.fsdp.ShardingStrategy.FULL_SHARD
        )
        
        # Mixed precision configuration
        mixed_precision = torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        
        # CPU offload configuration
        cpu_offload = None
        if fsdp_config.cpu_offload:
            cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True)
        
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
        )
    
    elif strategy_name == "deepspeed":
        # Basic DeepSpeed configuration
        config = {
            "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,  # You can make this configurable
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            }
        }
        
        return DeepSpeedStrategy(config=config)
    
    else:
        logger.warning(f"Unknown strategy: {strategy_name}. Using auto.")
        return "auto"


def setup_wandb_logger(args):
    """Setup Wandb logger based on wandb config"""
    
    if not hasattr(args, 'wandb') or not args.wandb.enable:
        return None
    
    wandb_config = args.wandb
    
    # Generate run name and notes
    curr_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_name}_{args.dataset_name}_{curr_time_str}"
    
    notes = wandb_config.notes
    if notes is None:
        notes = f"Fine-tuning {args.model_name} on {args.dataset_name} at {curr_time_str}"
    
    # Create experiment config for logging
    experiment_config = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "epochs": args.epochs,
        "peft": args.peft if hasattr(args, 'peft') else None,
        "lora_rank": getattr(args, 'lora_rank', None),
        "use_int4": getattr(args, 'use_int4', False),
        "use_int8": getattr(args, 'use_int8', False),
    }
    
    # Add distributed training config if available
    if hasattr(args, 'distributed_training') and args.distributed_training.use_distributed:
        experiment_config.update({
            "strategy": args.distributed_training.strategy,
            "num_devices": args.distributed_training.devices,
            "num_processes": args.distributed_training.num_processes,
        })
    
    return WandbLogger(
        project=wandb_config.project_name,
        name=run_name,
        notes=notes,
        config=experiment_config,
        mode=wandb_config.mode,
        log_model=wandb_config.log_model,
        save_code=wandb_config.save_code,
    )


def setup_model_and_tokenizer(args):
    """Setup model and tokenizer based on args"""
    if args.token is None:
        access_token = os.getenv("HF_TOKEN", None)
    else:
        access_token = args.token

    config = AutoConfig.from_pretrained(args.model_name,
                                        token=access_token,
                                        cache_dir=args.cache_path)

    config.use_cache = False
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]

    use_flash_attention = False

    if not args.disable_flash_attention and model_type != "llama":
        logger.info("Model is not llama, disabling flash attention...")
    elif args.disable_flash_attention and model_type == "llama":
        logger.info("Model is llama, could be using flash attention...")
    elif not args.disable_flash_attention and torch.cuda.get_device_capability()[0] >= 8:
        from ft_llms.llama_patch import replace_attn_with_flash_attn
        logger.info("Using flash attention for llama...")
        replace_attn_with_flash_attn()
        use_flash_attention = True

    # Setup tokenizer
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, token=access_token,
                                                  trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                  add_eos_token=args.add_eos_token, add_bos_token=args.add_bos_token,
                                                  use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token,
                                                  trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                  add_eos_token=args.add_eos_token, add_bos_token=args.add_bos_token,
                                                  use_fast=True)

    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization
    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        logger.info("Using no quantization")
        bnb_config = None

    # Setup PEFT config
    peft_config = None
    if not args.disable_peft:
        if args.peft == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
        elif args.peft == "prefix-tuing":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                num_virtual_tokens=args.p_tokens,
                encoder_hidden_size=args.p_hidden)
        elif args.peft == "p-tuing":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=args.p_tokens,
                encoder_hidden_size=args.p_hidden)
        elif args.peft == "ia3":
            peft_config = IA3Config(
                peft_type="IA3",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"],
            )

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Handle device mapping - disable for distributed strategies that don't support it
    kwargs = {}
    if args.split_model and not (hasattr(args, 'distributed_training') and 
                                args.distributed_training.use_distributed and 
                                args.distributed_training.strategy.lower() in ["fsdp", "deepspeed"]):
        logger.info("Splitting the model across all available devices...")
        kwargs["device_map"] = "auto"
    elif args.split_model:
        logger.warning("Disabling model splitting for distributed strategy compatibility")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        token=access_token, 
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code, 
        cache_dir=args.cache_path,
        torch_dtype=torch_dtype, 
        config=config,
        **kwargs
    )

    if use_flash_attention:
        from ft_llms.llama_patch import llama_forward_with_flash_attn
        assert model.model.layers[0].self_attn.forward.__doc__ == llama_forward_with_flash_attn.__doc__, \
            "Model is not using flash attention"

    if not args.disable_peft and peft_config is not None:
        logger.info("Using PEFT...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model)
            if use_flash_attention:
                from ft_llms.llama_patch import upcast_layer_for_flash_attention
                logger.info("Upcasting flash attention layers...")
                model = upcast_layer_for_flash_attention(model, torch_dtype)
        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)
    else:
        logger.info("Using Full Finetuning")

    print_trainable_parameters(model)
    
    return model, tokenizer


def main_llms_finetune(args_filename=None, args_dict=None):
    # Set tokenizer parallelism to False to avoid fork warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if args_filename is not None:
        assert os.path.exists(args_filename), f"Config file {args_filename} does not exist"
        tmp_args = parse_args(args_filename)
    else:
        args_filename = os.path.join(parent_dir_path, 'configs', 'llms_finetune_config.yaml')
        tmp_args = parse_args(args_filename)
    
    if args_dict is not None:
        assert isinstance(args_dict, dict), f"args_dict should be a dictionary"
        tmp_args.update_config_from_dict(args_dict)
        
    args = tmp_args

    # print the arguments being used only if on the main thread
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            args.print_config()
    else:
        args.print_config()
    
    # Handle debug mode and distributed training conflict
    if args.debug and hasattr(args, 'distributed_training') and args.distributed_training.use_distributed:
        print("\033[93mWarning: Debug mode is enabled but distributed training is also requested. Distributed training will take precedence.\033[0m")
        # Keep distributed training settings as-is
    elif args.debug:
        # Force disable distributed training only when debug is true AND distributed is not requested
        logger.warning("Debug mode enabled - disabling distributed training")
        if hasattr(args, 'distributed_training'):
            args.distributed_training.use_distributed = False
    
    # Setup Wandb environment
    if hasattr(args, 'wandb') and args.wandb.enable:
        # Wandb will be initialized by the logger
        pass
    else:
        # Disable wandb if not configured
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_SILENT"] = "true"
        logger.info("Wandb disabled.")

    # Handle model name mapping (preserve original logic)
    if hasattr(args, 'refer_data_source') and args.refer_data_source is not None:
        args.model_name = args.refer_data_source
    if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        args.model_name = "decapoda-research/llama-7b-hf"

    # Setup distributed training strategy
    strategy = get_training_strategy(args)
    
    # Determine devices
    devices = 1 
    if hasattr(args, 'distributed_training') and args.distributed_training.use_distributed:
        devices = args.distributed_training.devices
        if devices != "auto":
            devices = int(devices)
        logger.info(f"Using distributed training with {devices} devices and {args.distributed_training.strategy} strategy")
    else:
        logger.info("Using single device training")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Setup data module
    data_module = LLMDataModule(args, tokenizer)
    
    # Setup Lightning module
    lightning_module = LLMFineTuneModule(args, model, tokenizer)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{step}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            every_n_train_steps=None, # getattr(args, 'save_epochs', None) if hasattr(args, 'save_epochs') and args.save_epochs is not None else None,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=10), # refresh every 10 batches
    ]
         
    # Setup Wandb logger (only if enabled)
    wandb_logger = setup_wandb_logger(args)
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        strategy=strategy,
        precision="16-mixed" if not torch.cuda.is_bf16_supported() else "bf16-mixed",
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=getattr(args, 'log_steps', 50),
        callbacks=callbacks,
        logger=wandb_logger,  # Will be None if wandb is disabled
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        # deterministic=True if args.debug else False,
        # Distributed training optimizations - only enable if actually using distributed
        sync_batchnorm=True if (strategy != "auto" and not args.debug) else False,
        # use_distributed_sampler=True,
    )
    
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training strategy: {strategy}")
    logger.info(f"Devices: {devices}")
    if wandb_logger:
        logger.info("Wandb logging enabled")
    else:
        logger.info("Wandb logging disabled")

    # Train the model
    try:
        trainer.fit(lightning_module, data_module)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Clean up any distributed processes
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        trainer.fit(lightning_module, data_module)
    
    # Save the final model
    if not args.disable_peft:
        model = lightning_module.model.merge_and_unload()
    else:
        model = lightning_module.model
    
    # Create output directory
    create_folder(args.output_dir)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main_llms_finetune()