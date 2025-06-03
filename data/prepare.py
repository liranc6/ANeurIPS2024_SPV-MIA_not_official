import os
import random
import datasets
import trl
from attack.utils import create_folder

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None

def packing_texts(examples):
    more_examples = True
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"]
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    # print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    return {
        "text": packed_texts
    }
    
def select_dataset_indices(dataset, split_args, split_name="train"):
    """
    Selects a subset of the dataset based on start_idx and end_idx in split_args.
    """
    if hasattr(split_args, 'start_idx') and hasattr(split_args, 'end_idx'):
        end_idx = split_args.end_idx if split_args.end_idx > 0 else len(dataset)
        start_idx = split_args.start_idx if split_args.start_idx > 0 else 0
        end_idx = min(end_idx, len(dataset))
        start_idx = min(max(start_idx, 0), end_idx)
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"Selected {split_name} data from index {split_args.start_idx} to {end_idx} (total: {len(dataset)})")
    return dataset

def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    train_dataset = datasets.load_dataset(
        args.dataset.name,
        args.dataset.config_name,
        split=f"train[:{int((1-args.validation_split_percentage)*100)}%]"
    )
    valid_dataset = datasets.load_dataset(
        args.dataset.name,
        args.dataset.config_name,
        split=f"train[{int((1-args.validation_split_percentage)*100)}%:]",
    )

    # Apply dataset train/eval indices from config if provided
    if hasattr(args.dataset, 'train'):
        train_dataset = select_dataset_indices(train_dataset, args.dataset.train, split_name="train")
    if hasattr(args.dataset, 'eval'):
        valid_dataset = select_dataset_indices(valid_dataset, args.dataset.eval, split_name="evaluation")

    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"

    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if args.packing:
        global block_size, tokenizer_, max_buff_size
        block_size = args.block_size
        max_buff_size = block_size * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        create_folder(f"{args.cache_path}/{args.dataset.name}/{args.dataset.config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset.name}/{args.dataset.config_name}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset.name}/{args.dataset.config_name}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )

        # Re-apply dataset train/eval indices after packing if needed
        try:
            if args.debug:
                if hasattr(args.dataset, 'train'):
                    train_dataset = select_dataset_indices(train_dataset, args.dataset.train, split_name="train")
                if hasattr(args.dataset, 'eval'):
                    valid_dataset = select_dataset_indices(valid_dataset, args.dataset.eval, split_name="evaluation")
        except Exception as e:
            print(f"Error selecting dataset indices after packing: {e}")
            print("Skipping index selection after packing.")            

    return train_dataset, valid_dataset