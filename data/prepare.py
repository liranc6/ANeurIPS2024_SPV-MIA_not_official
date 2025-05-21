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
def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    # raw_datasets = datasets.load_dataset(args.dataset.name, args.dataset.config_name)['train']
    # if "validation" in raw_datasets.keys():
    #     train_dataset = raw_datasets["train"]
    #     valid_dataset = raw_datasets["validation"]
    # else:
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

    # train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
    # valid_idxs = set(range(len(raw_datasets))) - train_idxs
    # train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
    # valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])
    
    # Apply dataset train/eval indices from config if provided
    if hasattr(args.dataset, 'train') and hasattr(args.dataset.train, 'start_idx') and hasattr(args.dataset.train, 'end_idx'):
        end_idx = args.dataset.train.end_idx if args.dataset.train.end_idx > 0 else len(train_dataset)
        start_idx = args.dataset.train.start_idx if args.dataset.train.start_idx > 0 else 0
        train_dataset = train_dataset.select(range(start_idx, end_idx))
        print(f"Selected train data from index {args.dataset.train.start_idx} to {end_idx} (total: {len(train_dataset)})")
    
    if hasattr(args.dataset, 'eval') and hasattr(args.dataset.eval, 'start_idx') and hasattr(args.dataset.eval, 'end_idx'):
        end_idx = args.dataset.eval.end_idx if args.dataset.eval.end_idx > 0 else len(valid_dataset)
        start_idx = args.dataset.eval.start_idx if args.dataset.eval.start_idx > 0 else 0
        valid_dataset = valid_dataset.select(range(start_idx, end_idx))
        print(f"Selected evaluation data from index {args.dataset.eval.start_idx} to {end_idx} (total: {len(valid_dataset)})")

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
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset.name}/{args.dataset.config_name}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset.name}/{args.dataset.config_name}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        return train_dataset, valid_dataset