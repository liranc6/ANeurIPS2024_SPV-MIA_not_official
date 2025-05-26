import os
import sys
import time
import sys
import os
from weakref import ref
from regex import E, F
import numpy as np
import matplotlib.pyplot as plt

import time

# Check GPU availability
import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0)



here = os.path.dirname(__file__)
parent_dir_path = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(parent_dir_path)
sys.path.append(parent_dir_path)
sys.path.append(PROJECT_DIR)

# Change working directory to script location
os.chdir(here)

from src.parser import parse_args


def printIsFileExists(file_path):
    exist = os.path.exists(file_path)
    if exist:
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")
        
    return exist

def import_required_modules(args):
    finetune_script_path = args.finetune_script_path
    printIsFileExists(finetune_script_path)
    sys.path.append(os.path.dirname(finetune_script_path))
    from llms_finetune import main_llms_finetune
    
    refer_data_script_path = args.refer_data_script_path
    printIsFileExists(refer_data_script_path)
    sys.path.append(os.path.dirname(refer_data_script_path))
    from refer_data_generate import run_data_generation
    
    attack_script_path = args.attack_script_path
    printIsFileExists(attack_script_path)
    sys.path.append(os.path.dirname(attack_script_path))
    from run_attack import run_attack as run_attack_function

def plot_roc_curve(model_name, dataset_name, dataset_config_name, save_fig=True, show_fig=True):
    """
    Plot ROC curve for membership inference attack results.
    
    Args:
        model_name (str): Name of the model used for the attack
        dataset_name (str): Name of the dataset used
        dataset_config_name (str): Configuration name of the dataset
        save_fig (bool): Whether to save the figure to disk
        show_fig (bool): Whether to display the plot
        
    Returns:
        tuple: (AUC score, TPR at 1% FPR)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Load ROC curve data
    ROC_dir = f"./cache/{dataset_name}/{dataset_config_name}/attack_data_{model_name}@{dataset_name}"
    data_path = os.path.join(ROC_dir, "roc_stat.npz")
    
    if not os.path.exists(data_path):
        print(f"Error: ROC data not found at {data_path}")
        return None, None
        
    data = np.load(data_path)
    fpr = data["fpr"]
    tpr = data["tpr"]
    
    # Calculate AUC score
    auc_score = np.round(np.trapz(tpr, fpr), 4)
    
    # Find TPR at 1% FPR
    fpr_1_index = np.argmin(np.abs(fpr - 0.01))
    tpr_at_1pct_fpr = tpr[fpr_1_index]
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – Statistical Membership Inference Attack\n{model_name} on {dataset_name}")
    plt.legend()
    plt.grid(True)
    
    # Mark TPR@1%FPR
    plt.scatter(fpr[fpr_1_index], tpr[fpr_1_index], color='red', 
                label=f'TPR@1%FPR: {tpr_at_1pct_fpr:.3f}')
    plt.legend()
    
    # Save the figure if requested
    if save_fig:
        fig_path = os.path.join(ROC_dir, "roc_stat.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved to {fig_path}")
    
    # Show the figure if requested
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return auc_score, tpr_at_1pct_fpr

def run_experiment(override_args=None):
    config_file_relative_path = os.path.join(parent_dir_path, 'configs', 'pipeline_attack_config.yaml')
    printIsFileExists(config_file_relative_path)
    args = parse_args(config_file_relative_path)
    args.update_config_from_dict(override_args)
    
    model_name = args.model_name
    dataset_name = args.dataset.name
    dataset_config_name = args.dataset.config_name
    args.print_config()
    
    # import_required_modules
    finetune_script_path = args.finetune_script_path
    printIsFileExists(finetune_script_path)
    sys.path.append(os.path.dirname(finetune_script_path))
    from llms_finetune import main_llms_finetune
    
    refer_data_script_path = args.refer_data_script_path
    printIsFileExists(refer_data_script_path)
    sys.path.append(os.path.dirname(refer_data_script_path))
    from refer_data_generate import run_data_generation
    
    attack_script_path = args.attack_script_path
    printIsFileExists(attack_script_path)
    sys.path.append(os.path.dirname(attack_script_path))
    from run_attack import run_attack as run_attack_function
    
    
    if 'curr_time_str' in args.configs and args.curr_time_str is not None:
        curr_time_str = args.curr_time_str
    else:
        curr_time_str = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
                    
    
    # finetune target model
    finetune_config_path = args.finetune_config_path
    printIsFileExists(finetune_config_path)
    if args.debug:
        target_model_output_dir = os.path.join(".", "ft_llms", "debug", model_name, dataset_name, curr_time_str, "target")
    else:
        target_model_output_dir = os.path.join(".", "ft_llms", model_name, dataset_name, curr_time_str, "target")
        
    args.configs['target_model_output_dir'] = target_model_output_dir
    print(f"{target_model_output_dir=}")
    target_model_args = {
        "debug": args.debug,
        "output_dir": target_model_output_dir,
        "refer": False,
        "epochs": args.target_model_args.epochs,
        "model_name": model_name,
        "dataset": args.dataset,
        "curr_time_str": curr_time_str,
    }
    
    if os.path.exists(target_model_output_dir):
        print(f"Target model already exists at {target_model_output_dir}. Skipping finetuning.")
    else:
        main_llms_finetune(finetune_config_path, target_model_args)
    
    
    # create reference dataset
    data_generation_config_path = args.data_generation_config_path
    printIsFileExists(data_generation_config_path)
    if args.debug:
        generated_dataset_dir = os.path.join(args.cache_path, 'debug', args.dataset.name, args.dataset.config_name, f"refer@{args.model_name}", curr_time_str)
    else:
        generated_dataset_dir = os.path.join(args.cache_path, args.dataset.name, args.dataset.config_name, f"refer@{args.model_name}", curr_time_str)
        
    data_generation_args = {
        "debug": args.debug,
        "generated_dataset_dir": generated_dataset_dir,
        "model_name": model_name,
        "target_model": target_model_args['output_dir'],
        "dataset": args.dataset,
        "curr_time_str": curr_time_str,
    }

    if os.path.exists(generated_dataset_dir):
        print(f"Generated dataset already exists at {generated_dataset_dir}. Skipping data generation.")
    else:
        run_data_generation(data_generation_config_path, data_generation_args)
    
    # finetune reference model
    if args.debug:
        reference_model_output_dir = os.path.join(".", "ft_llms", "debug", model_name, dataset_name, curr_time_str, "reference")
    else:
        reference_model_output_dir = os.path.join(".", "ft_llms", model_name, dataset_name, curr_time_str, "reference")
        
    args.configs['reference_model_output_dir'] = reference_model_output_dir
    print(f"{reference_model_output_dir=}")
    reference_model_args = {
        "debug": args.debug,
        "output_dir": reference_model_output_dir,
        "refer": args.reference_model_args.refer,
        "epochs": args.reference_model_args.epochs,
        "model_name": model_name,
        "dataset": args.dataset,
        "curr_time_str": curr_time_str,
        "generated_dataset_dir": generated_dataset_dir,
    }
    
    if os.path.exists(reference_model_output_dir):
        print(f"Reference model already exists at {reference_model_output_dir}. Skipping finetuning.")
    else:
        main_llms_finetune(finetune_config_path, reference_model_args)
    
    attack_data_path = os.path.join(args.cache_path, args.dataset.name, args.dataset.config_name)
    if args.debug:
        args.mask_filling_model_name = model_name
        
    run_attack_args = {
        "debug": args.debug,
        "model_name": model_name,
        "dataset": args.dataset,
        "curr_time_str": curr_time_str,
        "target_model": target_model_args['output_dir'],
        "reference_model": reference_model_args['output_dir'],
        "cache_path": args.cache_path,
        "attack_type": args.attack_args.attack_type,
        "attack_data_path": attack_data_path,
        "mask_filling_model_name": args.mask_filling_model_name,
    }
    
    args.update_config_from_dict(run_attack_args)
    # run attack
    run_attack_function(None, args_dict=args.configs)
    
    # plot ROC curve
    auc_score, tpr_at_1pct_fpr = plot_roc_curve(model_name, dataset_name, dataset_config_name, save_fig=args.save_fig, show_fig=args.show_fig)
    print(f"AUC Score: {auc_score}")
    print(f"TPR at 1% FPR: {tpr_at_1pct_fpr}")
    
    results_csv_path = "/home/liranc6/W25/adversarial-attacks-on-deep-learning/project/ANeurIPS2024_SPV-MIA_not_official/exp_results.csv"
    with open(results_csv_path, "a") as f:
        f.write(f"{model_name},{dataset_name},{dataset_config_name},{args.attack_args.attack_type},{args.attack_args.peak_top_k},{auc_score},{tpr_at_1pct_fpr}\n")
    print(f"Results saved to {results_csv_path}")


if __name__ == "__main__":
    
    print("Running script: pipeline_attack.py")
    
    model_names = ['EleutherAI/gpt-j-6B'] #, []'gpt2', 'tiiuae/falcon-rw-1b', 'EleutherAI/gpt-j-6B'] #, 'meta-llama/Llama-2-7b-hf']
    dataset_names_and_configs = {'wikitext': 'wikitext-2-raw-v1',
                                # 'xsum': 'EdinburghNLP/xsum',
                                'ag_news': 'fancyzhx/ag_news',
                                }

    attack_types = ['SPV-MIA_split_to_words', 'ours', 'SPV-MIA_correct_split_to_tokens', ]
    peak_top_k = ['5', '10', '15', '19']

    clis = []
    override_args = []

    for model_name in model_names:
        for dataset_name, dataset_config_name in dataset_names_and_configs.items():
            for attack_type in attack_types:
                for k in peak_top_k:
                    # CLI command structure with proper nesting
                    command = f"--model_name \"{model_name}\" --dataset.name \"{dataset_name}\" --dataset.config_name \"{dataset_config_name}\" --attack_args.attack_type \"{attack_type}\" --attack_args.attack_strategy.peak_top_k {k}"
                    clis.append(command)
                    
                    # Dictionary structure with proper nesting
                    override_args.append({
                        'debug': False,
                        'model_name': model_name,
                        'dataset': {
                            'name': dataset_name,
                            'config_name': dataset_config_name
                        },
                        'attack_args': {
                            'attack_type': attack_type,
                            'attack_strategy': {
                                'peak_top_k': int(k)  # Convert to integer
                            }
                        }
                    })
                    
                    if attack_type != 'ours':
                        break
    
    for override_arg in override_args:
        try:
            print(f"Running with override args: {override_arg}")
            run_experiment(override_args=override_arg)
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    print("Pipeline attack completed.")

    

    