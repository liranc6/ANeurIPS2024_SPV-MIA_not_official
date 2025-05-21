import os
from pydoc import text
import random

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from attack import utils
from attack.utils import Dict
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm as original_tqdm
from torch.utils.data import DataLoader
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from itertools import cycle
import matplotlib.pyplot as plt
import re
import seaborn as sns
from functools import partial
import sys
import os

logger = get_logger(__name__, "INFO")

PATH = os.getcwd()
here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(here)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)

from src.utils import generate_neighbors


def tqdm(iterable, desc, position=0, leave=False, **kwargs):
    """Helper function to create consistent progress bars"""
    if 'dynamic_ncols' not in kwargs:
        kwargs['dynamic_ncols'] = True
        
    return original_tqdm(
        iterable,
        desc=desc,
        leave=leave,
        position=position,
        **kwargs,
        # disable=not accelerator.is_main_process
    )
    
accelerator = Accelerator()
class AttackModel:
    def __init__(self, target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=None, mask_tokenizer=None):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.kind = cfg.configs['attack_kind']
        self.cfg = cfg
        if mask_model is not None:
            self.mask_model = mask_model
            self.mask_tokenizer = mask_tokenizer
            self.pattern = re.compile(r"<extra_id_\d+>")
        if shadow_model is not None and cfg.configs['attack_kind'] == "nn":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model

    def llm_eval(self, model, data_loader, cfg, idx_rate, perturb_fn=None, refer_model=None):
        model.eval()
        losses = []
        ref_losses = []
        token_lens = []
        data_loader_pbar = tqdm(
            enumerate(data_loader),
            "Evaluating",
            # position=0,
            leave=True,
        )
        for iteration, texts in data_loader_pbar:
            texts = texts["text"]
            if cfg.configs["maximum_samples"] is not None:
                if iteration * accelerator.num_processes >= cfg.configs["maximum_samples"]:
                    break
            if perturb_fn is not None:
                texts = perturb_fn(texts)
                if texts is None or len(texts) == 0 or texts == "":
                    raise ValueError("No perturbed texts generated.")
            token_ids = self.tokenizer(texts, return_tensors="pt", padding=True).to(accelerator.device)
            labels = token_ids.input_ids
            with torch.no_grad():
                outputs = model(**token_ids, labels=labels)
                ref_outputs = refer_model(**token_ids, labels=labels)
            loss = outputs.loss
            ref_loss = ref_outputs.loss
            token_lens.append(accelerator.gather(torch.tensor(token_ids.input_ids.size()[-1]).reshape(-1, 1).to(accelerator.device)).detach().cpu().numpy()) # TODO: may cause bug when running attacks in paralell.
            losses.append(accelerator.gather(loss.reshape(-1, 1)).detach().cpu().to(torch.float32).numpy())
            ref_losses.append(accelerator.gather(ref_loss.reshape(-1, 1)).detach().cpu().to(torch.float32).numpy())
            # print(f"{accelerator.device}@{texts}")
            # print(f"time duration: {time.time() - start_time}s")
        losses = np.concatenate(losses, axis=0)
        ref_losses = np.concatenate(ref_losses, axis=0)
        token_lens = np.concatenate(token_lens, axis=0)
        # token_lens = np.array(token_lens, dtype=np.int32)
        return losses, ref_losses, token_lens

    def eval_perturb(self, model, dataset, cfg):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        
        Generate cfg.configs["perturbation_number"] perturbations for each text in the dataset.
        """
        per_losses = []
        ref_per_losses = []
        ori_losses = []
        ref_ori_losses = []
        ori_dataset = deepcopy(dataset)
        perturbation_pbar = tqdm(
                                range(cfg.configs["perturbation_number"]), 
                                "Generating perturbed samples",
                                # position=0,
                                leave=True
                                )
        
        for i in perturbation_pbar:
            idx_rate = i / cfg.configs["perturbation_number"] * 0.7 # perturbation rate
            ori_loss, ref_ori_loss, ori_token_len = self.llm_eval(model, ori_dataset, cfg, idx_rate, refer_model=self.reference_model)
            ori_losses.append(ori_loss)
            ref_ori_losses.append(ref_ori_loss)
            perturb_fn = partial(self.sentence_perturbation, idx_rate=idx_rate)
            sampled_per_losses = []
            sampled_ref_per_losses = []
            
            sample_pbar = tqdm(
                                range(cfg.configs["sample_number"]), 
                                "Generating perturbed samples", 
                                # position=1,
                                # leave=False
                            )
            for _ in sample_pbar: # generate multiple perturbed samples and get the losses
                per_loss, ref_per_loss, per_token_len = self.llm_eval(model, ori_dataset, cfg, idx_rate, perturb_fn=perturb_fn, refer_model=self.reference_model)
                sampled_per_losses.append(per_loss)
                sampled_ref_per_losses.append(ref_per_loss)
            sampled_per_losses = np.concatenate(sampled_per_losses, axis=-1)
            sampled_ref_per_losses = np.concatenate(sampled_ref_per_losses, axis=-1)
            per_losses.append(np.expand_dims(sampled_per_losses, axis=-1))
            ref_per_losses.append(np.expand_dims(sampled_ref_per_losses, axis=-1))
        ori_losses = np.concatenate(ori_losses, axis=-1)
        ref_ori_losses = np.concatenate(ref_ori_losses, axis=-1)
        per_losses = np.concatenate(per_losses, axis=-1)
        var_losses = per_losses - np.expand_dims(ori_losses, axis=-2)
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg.configs["calibration"] else None
        ref_var_losses = ref_per_losses - np.expand_dims(ref_ori_losses, axis=-2) if cfg.configs["calibration"] else None

        output = (Dict(
            per_losses=per_losses,
            ori_losses=ori_losses,
            var_losses=var_losses,
        ),
        Dict(
            ref_per_losses=ref_per_losses,
            ref_ori_losses=ref_ori_losses,
            ref_var_losses=ref_var_losses,
        ))
        return output

    def data_prepare(self, kind, cfg):
        """
        Prepare the data for the attack model.
        
        :param kind: Type of data (e.g., 'target', 'shadow')
        :param cfg: Configuration dictionary containing parameters for data preparation
        :return: A dictionary containing prepared data
        
        This function generates or loads feature vectors for member and non-member data.
        It checks if the feature vectors already exist in the specified path. If they do not exist or if
        the `load_attack_data` flag is set to False, it generates the feature vectors using the `eval_perturb` method.
        The generated feature vectors are saved to the specified path. If the feature vectors already exist
        and the `load_attack_data` flag is set to True, it loads the feature vectors from the specified path.
        The function also handles the case where calibration is required by saving and loading reference feature vectors.
        The function returns a dictionary containing the member feature vectors, non-member feature vectors,
        reference member feature vectors, and reference non-member feature vectors.
        """
        logger.info("Preparing data...")
        data_path = os.path.join(PATH, cfg.configs["attack_data_path"], f"attack_data_{cfg.configs['model_name']}@{cfg.configs['dataset_name']}")
        target_model = getattr(self, kind + "_model")
        mem_data = self.datasets[kind]["train"]
        nonmem_data = self.datasets[kind]["valid"]

        mem_path = os.path.join(data_path, kind, "mem_feat.npz")
        nonmem_path = os.path.join(data_path, kind, "nonmen_feat.npz")
        ref_mem_path = os.path.join(data_path, kind, "ref_mem_feat.npz")
        ref_nonmem_path = os.path.join(data_path, kind, "ref_nonmen_feat.npz")

        pathlist = (mem_path, nonmem_path, ref_mem_path, ref_nonmem_path) if cfg.configs["calibration"] else (mem_path, nonmem_path)

        if not utils.check_files_exist(*pathlist) or not cfg.configs["load_attack_data"]:
            with open(cfg.closest_tokens_path, 'rb') as f:
                self.cosine_similarities = torch.load(f)
                assert isinstance(self.cosine_similarities, dict), "cosine_similarities should be a dict"
                
            logger.info("Generating feature vectors for member data...")
            mem_feat, ref_mem_feat = self.eval_perturb(target_model, mem_data, cfg)
            if accelerator.is_main_process:
                utils.save_dict_to_npz(mem_feat, mem_path)
                if cfg.configs["calibration"]:
                    utils.save_dict_to_npz(ref_mem_feat, ref_mem_path)

            logger.info("Generating feature vectors for non-member data...")
            nonmem_feat, ref_nonmem_feat = self.eval_perturb(target_model, nonmem_data, cfg)
            if accelerator.is_main_process:
                utils.save_dict_to_npz(nonmem_feat, nonmem_path)
                if cfg.configs["calibration"]:
                    utils.save_dict_to_npz(ref_nonmem_feat, ref_nonmem_path)

            logger.info("Saving feature vectors...")

        else:
            logger.info("Loading feature vectors...")
            mem_feat = utils.load_dict_from_npz(mem_path)
            ref_mem_feat = utils.load_dict_from_npz(ref_mem_path) if cfg.configs["calibration"] else None
            nonmem_feat = utils.load_dict_from_npz(nonmem_path)
            ref_nonmem_feat = utils.load_dict_from_npz(ref_nonmem_path) if cfg.configs["calibration"] else None

        logger.info("Data preparation complete.")

        return Dict(
            mem_feat=mem_feat,
            nonmem_feat=nonmem_feat,
            ref_mem_feat=ref_mem_feat,
            ref_nonmem_feat=ref_nonmem_feat,
                    )

    def feat_prepare(self, info_dict, cfg):
        # mem_info = info_dict.mem_feat
        # ref_mem_info = info_dict.ref_mem_feat
        if cfg.configs["calibration"]:
            get_prob = lambda logprob: np.power(np.e, -logprob)
            mem_feat = ((get_prob(info_dict.mem_feat.per_losses).mean((-1, -2)) - get_prob(info_dict.mem_feat.ori_losses).mean(-1)) -
                        (get_prob(info_dict.ref_mem_feat.ref_per_losses).mean((-1, -2)) - get_prob(info_dict.ref_mem_feat.ref_ori_losses).mean(-1)))
            nonmem_feat = ((get_prob(info_dict.nonmem_feat.per_losses).mean((-1, -2)) - get_prob(info_dict.nonmem_feat.ori_losses).mean(-1)) -
                           (get_prob(info_dict.ref_nonmem_feat.ref_per_losses).mean((-1, -2)) - get_prob(info_dict.ref_nonmem_feat.ref_ori_losses).mean(-1)))
        else:
            mem_feat = info_dict.mem_feat.var_losses / info_dict.mem_feat.ori_losses
            nonmem_feat = info_dict.nonmem_feat.var_losses / info_dict.nonmem_feat.ori_losses


        if cfg.configs["attack_kind"] == "stat":
            # mem_feat = mem_feat[:, :, 0]
            # nonmem_feat = nonmem_feat[:, :, 0]
            # mem_feat[np.isnan(mem_feat)] = 0
            # nonmem_feat[np.isnan(nonmem_feat)] = 0
            feat = - np.concatenate([mem_feat, nonmem_feat])
            ground_truth = np.concatenate([np.zeros(mem_feat.shape[0]), np.ones(nonmem_feat.shape[0])]).astype(int)

        return feat, ground_truth

    def conduct_attack(self, cfg):
        save_path = os.path.join(PATH, cfg.configs["attack_data_path"], f"attack_data_{cfg.configs['model_name']}@{cfg.configs['dataset_name']}",
                                 f"roc_{cfg.configs['attack_kind']}.npz")

        raw_info = self.data_prepare("target", cfg)
        feat, ground_truth = self.feat_prepare(raw_info, cfg)
        # self.distinguishability_plot(raw_info['mem_feat']['ori_losses'].mean(-1),
        #                              raw_info['nonmem_feat']['ori_losses'].mean(-1))
        # self.distinguishability_plot(feat[:1000], feat[-1000:])
        self.eval_attack(ground_truth, -feat, path=save_path)

    def tokenize_and_mask(self, text, span_length, pct, idx_rate, ceil_pct=False):
        cfg = self.cfg
        
        attack_type = cfg.attack_type
        if attack_type == "ours":
            attack_strategy = cfg.attack_strategy
        
            tokens = self.mask_tokenizer.tokenize(text) # expectation: returns a list of tokens
        elif attack_type == "SPV-MIA_split_to_words":
            tokens = text.split(' ')
            
        mask_string = '<<<mask>>>'
        perturb_start_idx = int(len(tokens) * idx_rate)

        n_spans = pct * len(tokens) / (span_length + cfg.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        if attack_type == "ours":
            indexes_to_replace = np.random.choice(len(tokens), size=n_spans, replace=False)
            for idx in indexes_to_replace:
                tokens[idx] = mask_string
        else: # not attack_strategy == "ours": 
            while n_masks < n_spans:
                start = np.random.randint(0, len(tokens) - span_length)
                end = start + span_length
                search_start = max(0, start - cfg.buffer_size)
                search_end = min(len(tokens), end + cfg.buffer_size)
                if mask_string not in tokens[search_start:search_end]:
                    tokens[start:end] = [mask_string]
                    n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
            elif attack_type == 'ours' or attack_type == 'theirs_with_correct_split_to_tokens':
                tokens[idx] = self.mask_tokenizer.convert_tokens_to_string([token])
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    @staticmethod
    def count_masks(texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, texts):
        cfg = self.cfg
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(accelerator.device)

        if cfg.attack_type == "ours":
            closest_tokens_path = cfg.closest_tokens_path
            #load dict from file
            with open(closest_tokens_path, 'r') as f:
                closest_tokens = torch.load(f)
                assert isinstance(closest_tokens, dict), "closest_tokens should be a dict"
            # get the closest tokens for each token in the input
            
            ##TODO
            raise NotImplementedError("closest_tokens is not implemented yet")
            
        else:
            outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=cfg.mask_top_p,
                                      num_return_sequences=1, eos_token_id=stop_id)
            outputs = self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return outputs

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def sentence_perturbation(self, texts, idx_rate):
        """
        Perturb the input texts by replacing certain tokens with mask tokens.
        Generates one perturbed text for each input text.
        :param texts: List of input texts to be perturbed
        :param idx_rate: The rate at which to apply the perturbation
        :return: List of perturbed texts
        This function tokenizes the input texts, applies masking based on the specified rate,
        and replaces the masked tokens with generated fills from the model.
        It handles the case where the model does not generate the expected number of fills
        by retrying the process until all texts are filled.
        The function also ensures that the perturbed texts are returned in the same order as the input texts.
        The function uses the mask model to generate fills for the masked tokens.
        The function also handles the case where the model does not generate the expected number of fills
        """
        
        cfg = self.cfg
        
        strategy = cfg.attack_args.attack_strategy
        # strategy = {"name": "embeddings", 'peak_top_k': 4, 'max_neighbors': 2, 'n_tokens': 5}
        span_length = cfg.span_length
        buffer_size = cfg.buffer_size
        pct = cfg.pct
        multiplier = pct / (span_length + buffer_size * 2)
        
        perturbed_texts = []
        if cfg.attack_args.attack_type == "ours":
            single_text = texts
            tokens = self.mask_tokenizer.tokenize(single_text)
            n_spans = int(multiplier * len(tokens))
            strategy['n_tokens'] = n_spans
            
            neighbors, _ = generate_neighbors(single_text, None, self.mask_tokenizer, self.cosine_similarities, top_k=0, strategy=strategy)
            neighbor = neighbors[1]
            neighbor = neighbor['text']
            # original_text = neighbors.pop(0)
            # neighbor = neighbors[0]
            perturbed_texts.append(neighbor)
            
            # texts_pbar = tqdm(
            #                     enumerate(texts),
            #                     "Generating perturbed texts", 
            #                     # position=2,  # This should be properly positioned based on nesting level
            #                     leave=False,
            #                     total=len(texts),
            #                 )
            # # tqdm(enumerate(texts), total=len(texts), desc="Generating perturbed texts", leave=False)    
            # for idx, text in texts_pbar:
            #     tokens = self.mask_tokenizer.tokenize(text)
            #     n_spans = int(multiplier * len(tokens))
            #     strategy['n_tokens'] = n_spans
                
            #     neighbors, _ = generate_neighbors(text, None, self.mask_tokenizer, self.cosine_similarities, top_k=0, strategy=strategy)
            #     neighbor = neighbors[1]
            #     neighbor = neighbor['text']
            #     # original_text = neighbors.pop(0)
            #     # neighbor = neighbors[0]
            #     perturbed_texts.append(neighbor)
                
        else:       
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, idx_rate, cfg.ceil_pct) for x in texts]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            while '' in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
                print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
                masked_texts = [self.tokenize_and_mask(x, cfg.span_length, cfg.pct, idx_rate, cfg.ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        return perturbed_texts

    @staticmethod
    def eval_attack(y_true, y_scores, plot=True, path=None):
        if type(y_true) == torch.Tensor:
            y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        if path is not None:
            np.savez(path, fpr=fpr, tpr=tpr)
        auc_score = roc_auc_score(y_true, y_scores)
        logger.info(f"AUC on the target model: {auc_score}")

        # Finding the threshold point where FPR + TPR equals 1
        threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        logger.info(f"ASR on the target model: {threshold_point}")

        # Finding the threshold point where FPR + TPR equals 1
        tpr_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
        logger.info(f"TPR@1%FPR on the target model: {tpr_1fpr}")


        if plot:
            # plot the ROC curve
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score}; ASR = {threshold_point})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            # plot the no-skill line for reference
            plt.plot([0, 1], [0, 1], linestyle='--')
            # show the plot
            plt.show()
