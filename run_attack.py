import os
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import logging
import random
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict

import yaml
import datasets
from datasets import Image, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import sys
import os

here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(here)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)

from src.parser import parse_args

logger = logging.getLogger(__name__)

@rank_zero_only
def log_info(message):
    """Helper to log only on rank 0"""
    logger.info(message)
    
@rank_zero_only
def log_warning(message):
    """Helper to log warnings only on rank 0 with yellow color"""
    # ANSI color codes: yellow text
    yellow = '\033[93m'
    reset = '\033[0m'
    logger.warning(f"{yellow}{message}{reset}")

def run_attack(args_filename=None, args_dict=None):
    
    if args_filename is not None:
        assert os.path.exists(args_filename), f"Config file {args_filename} does not exist"
        tmp_args = parse_args(args_filename)
    else:
        args_filename = os.path.join(here, 'configs', 'config.yaml')
        tmp_args = parse_args(args_filename)
        
    if args_dict is not None:
        assert isinstance(args_dict, dict), f"args_dict should be a dictionary"
        tmp_args.update_config_from_dict(args_dict)
        
    cfg = tmp_args

    # print the arguments being used
    cfg.print_config()

    # Setup Lightning trainer for distributed coordination
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load abs path
    PATH = os.path.dirname(os.path.abspath(__file__))

    # Fix the random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = False

    ## Load generation models.
    if not cfg.load_attack_data:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Try to load locally first, then fallback to online download
        try:
            target_model = AutoModelForCausalLM.from_pretrained(
                cfg.target_model, 
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch_dtype,
                local_files_only=True,
                config=AutoConfig.from_pretrained(cfg.target_model, local_files_only=True),
                cache_dir=cfg.cache_path,
                trust_remote_code=True,
            )
        except (OSError, FileNotFoundError) as e:
            log_warning(f"Failed to load target model from local cache: {e}")
            log_warning("Attempting to download model from Hugging Face...")
            try:
                target_model = AutoModelForCausalLM.from_pretrained(
                    cfg.target_model, 
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    torch_dtype=torch_dtype,
                    local_files_only=False,  # Allow download
                    cache_dir=cfg.cache_path,
                    trust_remote_code=True,
                )
            except Exception as download_error:
                log_info(f"Failed to download target model: {download_error}")
                raise download_error
        
        try:
            reference_model = AutoModelForCausalLM.from_pretrained(
                cfg.reference_model, 
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch_dtype,
                local_files_only=True,
                config=AutoConfig.from_pretrained(cfg.model_name),
                cache_dir=cfg.cache_path,
                trust_remote_code=True,
            )
        except (OSError, FileNotFoundError) as e:
            log_warning(f"Failed to load reference model from local cache: {e}")
            log_warning("Attempting to download reference model from Hugging Face...")
            try:
                reference_model = AutoModelForCausalLM.from_pretrained(
                    cfg.reference_model, 
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    torch_dtype=torch_dtype,
                    local_files_only=False,  # Allow download
                    config=AutoConfig.from_pretrained(cfg.model_name),
                    cache_dir=cfg.cache_path,
                    trust_remote_code=True,
                )
            except Exception as download_error:
                log_info(f"Failed to download reference model: {download_error}")
                raise download_error

        log_info("Successfully load models")
        config = AutoConfig.from_pretrained(cfg.model_name)
        
        # Load tokenizer.
        model_type = config.model_type
        if model_type == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(
                cfg.model_name, 
                add_eos_token=cfg.add_eos_token,
                add_bos_token=cfg.add_bos_token, 
                use_fast=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, 
                add_eos_token=cfg.add_eos_token,
                add_bos_token=cfg.add_bos_token, 
                use_fast=True
            )

        if cfg.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
            cfg.model_name = "decapoda-research/llama-7b-hf"

        if cfg.pad_token_id is not None:
            log_info(f"Using pad token id {cfg.pad_token_id}")
            tokenizer.pad_token_id = cfg.pad_token_id

        if tokenizer.pad_token_id is None:
            log_info("Pad token id is None, setting to eos token id...")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load datasets - keep as raw datasets, not DataLoaders
        train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
        
        cfg.maximum_train_samples = min(cfg.maximum_samples, len(train_dataset["text"])) - 1
        train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg.maximum_train_samples)])
        cfg.maximum_val_samples = min(cfg.maximum_samples, len(valid_dataset["text"])) - 1
        valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg.maximum_val_samples)])
        log_info("Successfully load datasets!")

        # Load Mask filling model
        shadow_model = None
        int8_kwargs = {}
        half_kwargs = {}
        if cfg.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif cfg.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
            
        if 'SPV' in cfg.attack_type:
            print("using t5-base as a mask filling model")
            cfg.mask_filling_model_name = "t5-base"
            
        # Don't assign device here - Lightning will handle it
        if cfg.attack_type == "ours":
            mask_model = AutoModelForCausalLM.from_pretrained(
                cfg.mask_filling_model_name, 
                **int8_kwargs, 
                **half_kwargs
            )
        else:
            mask_model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.mask_filling_model_name, 
                **int8_kwargs, 
                **half_kwargs
            )
            
        try:
            # Try different possible attribute names for maximum sequence length
            if hasattr(mask_model.config, 'n_positions'):
                n_positions = mask_model.config.n_positions
            elif hasattr(mask_model.config, 'max_position_embeddings'):
                n_positions = mask_model.config.max_position_embeddings
            elif hasattr(mask_model.config, 'max_sequence_length'):
                n_positions = mask_model.config.max_sequence_length
            elif hasattr(mask_model.config, 'seq_length'):
                n_positions = mask_model.config.seq_length
            elif hasattr(mask_model.config, 'max_seq_len'):
                n_positions = mask_model.config.max_seq_len
            else:
                log_warning(f"Could not find sequence length attribute in {type(mask_model.config).__name__}")
                n_positions = 512  # Default fallback
        except AttributeError:
            log_warning("Error accessing mask model config, using default sequence length")
            n_positions = 512
            
        mask_tokenizer = AutoTokenizer.from_pretrained(
            cfg.mask_filling_model_name, 
            model_max_length=n_positions
        )

        # Note: No need to prepare dataloaders with accelerator.prepare()
        # AttackModel (as LightningModule) will handle distributed data loading internally
        
    else:
        target_model = None
        reference_model = None
        shadow_model = None
        mask_model = None
        train_dataset = None
        valid_dataset = None
        tokenizer = None
        mask_tokenizer = None

    # Pass raw datasets to AttackModel (not DataLoaders)
    datasets = {
        "target": {
            "train": train_dataset,
            "valid": valid_dataset
        }
    }

    # Create Lightning module
    attack_model = AttackModel(
        target_model, 
        tokenizer, 
        datasets, 
        reference_model, 
        shadow_model, 
        cfg, 
        mask_model=mask_model, 
        mask_tokenizer=mask_tokenizer
    )

    # Now conduct attack using Lightning's distributed capabilities
    # The AttackModel (as LightningModule) will use Lightning internally for distributed processing
    ROC_dir = attack_model.conduct_attack(cfg=cfg)
    
    return ROC_dir
    
if __name__ == "__main__":
    run_attack()