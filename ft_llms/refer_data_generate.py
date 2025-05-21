import os
import numpy as np
import torch
from tqdm.auto import tqdm as original_tqdm
from torch.utils.data import DataLoader
import logging
import random
import sys
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
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer


from src.parser import parse_args

def tqdm(*args, **kwargs):
    if 'dynamic_ncols' not in kwargs:
        kwargs['dynamic_ncols'] = True
    return original_tqdm(*args, **kwargs)

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
    
    # print the arguments being used
    args.print_config()
    
    # Create an accelerator for this run
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

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
        local_files_only=True,  # Consider setting to False if not found locally
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
    prompt_dataset = train_dataset# Dataset.from_dict(train_dataset)
    prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)

    model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

    # Generate texts based on prompts
    generated_dataset = {"text": []}
    print("Generating texts...")
    
    for i, text in enumerate(tqdm(prompt_dataloader)):
        try:
            prompt = (text["text"])
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(accelerator.device)
            clipped_ids = input_ids[:, :16]
            
            if hasattr(model, "module"):
                gen_tokens = model.module.generate(
                    clipped_ids,
                    num_beams=1,
                    do_sample=True,
                    max_length=input_ids.size(-1),
                )
            else:
                gen_tokens = model.generate(
                    clipped_ids,
                    num_beams=1,
                    do_sample=True,
                    max_length=input_ids.size(-1),
                )
                
            if model_type == "llama":
                gen_tokens = gen_tokens[:, 1:]
                
            loss = model(gen_tokens, labels=gen_tokens).loss
            # print(f"Sample {i}, Loss: {loss.item()}")
            
            gen_text = tokenizer.batch_decode(gen_tokens)
            generated_dataset["text"].extend(gen_text)
            
            # Periodically print a sample
            if i % 200 == 0:
                print(f"Sample generated text: {gen_text[0][:100]}...")
                
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue
        

    
    # Save the generated dataset
    generated_dataset = Dataset.from_dict(generated_dataset)
    
    # Handle special model name case
    if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        args.model_name = "decapoda-research/llama-7b-hf"
    
        
    save_dir = args.generated_dataset_dir #os.path.join(args.cache_path, args.dataset_name, args.dataset.config_name, f"refer@{args.model_name}", args.curr_time_str)
    print(f"Saving generated dataset to {save_dir}{accelerator.device}")
    generated_dataset.save_to_disk(os.path.join(save_dir, f"{accelerator.device}"))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Main process: concatenating datasets from all devices")
        concatenated_dataset = None
        for sub_dir in os.listdir(save_dir):
            data_path = os.path.join(save_dir, sub_dir)
            if os.path.isdir(data_path):
                print(f"Loading dataset from {data_path}")
                if concatenated_dataset is None:
                    concatenated_dataset = load_from_disk(data_path)
                else:
                    dataset = load_from_disk(data_path)
                    concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
                    
        print(f"Saving final concatenated dataset to {save_dir}")
        concatenated_dataset.save_to_disk(save_dir)
        print(f"Final dataset size: {len(concatenated_dataset)}")
    
    return True

if __name__ == "__main__":
    run_data_generation()