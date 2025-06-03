import os
import numpy as np
import torch
from tqdm.auto import tqdm as original_tqdm
from torch.utils.data import DataLoader, DistributedSampler
import logging
import random
import sys
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch.distributed as dist  # Import for explicit distributed control

here = os.path.dirname(__file__)
here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(here)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)
from data.prepare import dataset_prepare
from attack.utils import Dict
import argparse
import yaml
import datasets
from datasets import Image, Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer

from src.parser import parse_args

def tqdm(*args, **kwargs):
    if 'dynamic_ncols' not in kwargs:
        kwargs['dynamic_ncols'] = True
    return original_tqdm(*args, **kwargs)


class TextGenerationModule(L.LightningModule):
    """Lightning module for text generation"""
    
    def __init__(self, model, tokenizer, model_type):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def predict_step(self, batch, batch_idx):
        try:
            prompt = batch["text"]
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device)
            clipped_ids = input_ids[:, :16]
            
            gen_tokens = self.model.generate(
                clipped_ids,
                num_beams=1,
                do_sample=True,
                max_length=input_ids.size(-1),
            )
                
            if self.model_type == "llama":
                gen_tokens = gen_tokens[:, 1:]
                
            loss = self.model(gen_tokens, labels=gen_tokens).loss
            
            gen_text = self.tokenizer.batch_decode(gen_tokens)
            
            return {
                'generated_text': gen_text,
                'loss': loss.item(),
                'batch_idx': batch_idx
            }
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            return {
                'generated_text': [],
                'loss': None,
                'batch_idx': batch_idx
            }


def run_data_generation(args_filename=None, args_dict=None):
    
    if args_filename is not None:
        assert os.path.exists(args_filename), f"Config file {args_filename} does not exist"
        tmp_args = parse_args(args_filename)
    else:
        args_filename = os.path.join(parent_dir_path, 'configs', 'refer_data_generate_config.yaml')
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
        devices = -1  # Use all available GPUs
        strategy = "ddp"  # Distributed Data Parallel
    elif args.debug:
        print("Debug mode enabled - using single device for reference data generation")
        devices = 1
        strategy = "auto"
    else:
        devices = -1  # Use all available GPUs
        strategy = "ddp"  # Distributed Data Parallel
        
    # Setup Lightning trainer for multi-GPU
    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,  # Use specified devices
        strategy=strategy,  # Use specified strategy
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    
    print(f"Using {trainer.num_devices} devices")
    print(f"Strategy: {trainer.strategy}")
        

    # Initialize model, tokenizer, datasets
    config = AutoConfig.from_pretrained(args.model_name)
    config.use_cache = False
    bnb_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Add debugging print statements
    print(f"Loading model from: {args.target_model}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, 
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        local_files_only=False,
        config=config,
        cache_dir=args.cache_path
    )
    
    model_type = config.to_dict()["model_type"]
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token_id is None:
        print("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load datasets
    print("Preparing datasets...")
    train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Take a subset for prompt generation
    prompt_dataset = train_dataset
    
    # Lightning will automatically handle data distribution across GPUs
    prompt_dataloader = DataLoader(
        prompt_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=2  # Optional: add workers for data loading
    )

    # Create Lightning module
    lightning_module = TextGenerationModule(model, tokenizer, model_type)

    # Generate texts based on prompts
    print("Generating texts...")
    
    # Lightning automatically distributes prediction across all GPUs
    predictions = trainer.predict(lightning_module, prompt_dataloader)
    
    # Process results - Lightning automatically gathers results from all GPUs
    generated_dataset = {"text": []}
    
    # add a barrier to ensure all processes are synchronized before processing results
    if dist.is_initialized():
        dist.barrier()
        
    # add a barrier by lightning to ensure all processes are synchronized before processing results
    print("Processing generated texts...")
    # Only process on rank 0 to avoid duplication
    if trainer.is_global_zero:
        for i, batch_result in enumerate(predictions):
            if isinstance(batch_result, list):
                for result in batch_result:
                    if result and 'generated_text' in result and result['generated_text']:
                        generated_dataset["text"].extend(result['generated_text'])
                        
                        # Periodically print a sample
                        if 'batch_idx' in result and result['batch_idx'] % 200 == 0:
                            print(f"Sample generated text: {result['generated_text'][0][:100]}...")
            else:
                if batch_result and 'generated_text' in batch_result and batch_result['generated_text']:
                    generated_dataset["text"].extend(batch_result['generated_text'])
                    
                    # Periodically print a sample
                    if 'batch_idx' in batch_result and batch_result['batch_idx'] % 200 == 0:
                        print(f"Sample generated text: {batch_result['generated_text'][0][:100]}...")

    # Save the generated dataset - only on rank 0
    if trainer.is_global_zero:
        generated_dataset = Dataset.from_dict(generated_dataset)
        
        # Handle special model name case
        if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
            args.model_name = "decapoda-research/llama-7b-hf"
        
        save_dir = args.generated_dataset_dir
        print(f"Saving generated dataset to {save_dir}")
        generated_dataset.save_to_disk(save_dir)
        print(f"Final dataset size: {len(generated_dataset)}")
    
    return True

if __name__ == "__main__":
    run_data_generation()