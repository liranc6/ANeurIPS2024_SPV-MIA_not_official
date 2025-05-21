# import argparse

# import datasets
# import trl
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import torch
import logging
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config
import pandas as pd
import sys
here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(here)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)
from data.prepare import dataset_prepare
from attack.utils import create_folder
from transformers import LlamaTokenizer, get_scheduler
import os


from ft_llms.utils import constantlengthdatasetiter, print_trainable_parameters
# trl.trainer.ConstantLengthDataset.__dict__["__iter__"] = constantlengthdatasetiter
# setattr(trl.trainer.ConstantLengthDataset, "__iter__", constantlengthdatasetiter)

logger = logging.getLogger("finetune")

from src.parser import parse_args

def main_llms_finetune(args_filename=None, args_dict=None):
    
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

    # print the arguments being used
    args.print_config()
    
    accelerator = Accelerator()

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


    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_SILENT"] = "true"
        logger.info("Wandb disabled by default.")

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

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
    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        optimizer = "adamw_bnb_8bit"
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

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
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token, quantization_config=bnb_config,
                                                 trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                 torch_dtype=torch_dtype, config=config, **kwargs)

    if use_flash_attention:
        from ft_llms.llama_patch import llama_forward_with_flash_attn

        assert model.model.layers[
                   0].self_attn.forward.__doc__ == llama_forward_with_flash_attn.__doc__, "Model is not using flash attention"

    if not args.disable_peft:
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

    if args.refer_data_source is not None:
        args.model_name = args.refer_data_source
    if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        args.model_name = "decapoda-research/llama-7b-hf"
    with accelerator.main_process_first():
        train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)
        if args.refer:
            train_dataset = None
            # if args.debug:
            #     refer_data_path = os.path.join(args.cache_path, 'debug', args.dataset.name, args.dataset.config_name, f"refer@gpt2")
            # else:
            refer_data_path = args.generated_dataset_dir
            train_dataset = load_from_disk(refer_data_path)
            
        if args.debug:
            args.dataset.eval.end_idx = 30
            args.dataset.train.end_idx = 30
        train_dataset = Dataset.from_dict(train_dataset[args.dataset.train.start_idx:args.dataset.train.end_idx])
        valid_dataset = Dataset.from_dict(valid_dataset[args.dataset.eval.start_idx:args.dataset.eval.end_idx])
        # train_dataset = load_from_disk("/mnt/data0/fuwenjie/MIA-LLMs/cache/ag_news/None/refer@gpt2")

    logger.info(f"Training with {Accelerator().num_processes} GPUs")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_epochs,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        adam_epsilon=1e-6,
        report_to="none",
        logging_dir="./logs",
        load_best_model_at_end=False,
        save_total_limit=args.save_limit,
        bf16=False, #torch.cuda.is_bf16_supported(),
        fp16=True, #not torch.cuda.is_bf16_supported(),
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(valid_dataset)}")
    logger.info(f"Output directory: {args.output_dir}")

    # get trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="text",
    )
    trainer.tokenizer = tokenizer

    logger.info("Starting training with config:")
    logger.info(training_args)

    # train
    trainer.train()

    if not args.disable_peft:
        model = model.merge_and_unload()

    logger.info("Saving model to output_dir...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    
if __name__ == "__main__":
    # ANeurIPS2024_SPV-MIA_not_official/configs/llms_finetune_config.yaml
    
    main_llms_finetune()
