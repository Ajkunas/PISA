import json
import os
import sys
import pandas as pd
import warnings
from model_trainer import ModelTrainer
from sequence_creator import SequenceCreator
from utils import plot_truncate

warnings.filterwarnings("ignore")

def load_data(data_path):
    return pd.read_csv(data_path)

def create_sequence_creator(data, args_seq):
    return SequenceCreator(data=data, **args_seq)

def create_model_trainer(sequence_creator, args_model):
    return ModelTrainer(sequence_creator=sequence_creator, **args_model)

def load_existing_results(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_results(output_file, data):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def update_results(existing_data, experiment_name, args_seq, args_model, results):
    existing_data[experiment_name] = {"args_seq": args_seq, "args_model": args_model, "results": results}
    return existing_data

def run_experiment(experiment_name, data_path, args_seq, args_model, output_file):
    data = load_data(data_path)
    sequence_creator = create_sequence_creator(data, args_seq)
    model_trainer = create_model_trainer(sequence_creator, args_model)
    results = model_trainer.cross_validate()
    args_seq['data'] = data_path.split("/")[-1]

    existing_data = load_existing_results(output_file)
    updated_data = update_results(existing_data, experiment_name, args_seq, args_model, results)
    save_results(output_file, updated_data)

    print(f"Results for {experiment_name} successfully appended to {output_file}")

# only for truncation experiments
def run_plot_experiment_truncation(experiment_name, data_path, args_seq, args_model, plot_folder):
    data = load_data(data_path)
    sequence_methods = ["truncate", "truncate_subsequences"]
    modes = ["first", "last"]
    method_params = [1, 2, 3, 4, 5, 6, 7, 8]
    results = {}
    
    baseline_creator = create_sequence_creator(data, args_seq)
    baseline_trainer = create_model_trainer(baseline_creator, args_model)
    baseline_results = baseline_trainer.cross_validate()

    for sequence_method in sequence_methods:
        mean_aucs, std_aucs = {}, {}
        for mode in modes:
            mean_auc_mode, std_auc_mode = [], []
            for n in method_params:
                args_seq.update({"sequence_method": sequence_method, "method_mode": mode, "method_param": n})
                sequence_creator = create_sequence_creator(data, args_seq)
                model_trainer = create_model_trainer(sequence_creator, args_model)
                results = model_trainer.cross_validate()
                mean_auc_mode.append(results[0]["Mean AUC"])
                std_auc_mode.append(results[0]["Standard Deviation of AUC"])
            
            mean_aucs[sequence_method] = mean_aucs.get(sequence_method, {})
            std_aucs[sequence_method] = std_aucs.get(sequence_method, {})
            mean_aucs[sequence_method][mode] = mean_auc_mode
            std_aucs[sequence_method][mode] = std_auc_mode
        
        # plot results
        plot_truncate(sequence_method, method_params, mean_aucs, std_aucs, baseline_results[0]["Mean AUC"], plot_folder)

def main():
    experiment_type = sys.argv[1]
    data_path = sys.argv[2]
    output_file = sys.argv[3]
    output_path = f"results/{output_file}"

    experiments = {
        'truncation': [
                    {"experiment_name": "experiment1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    ],
        'correlation': [
                    # {"experiment_name": "experiment1-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment1-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment1-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment1-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment1-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment2-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment2-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment2-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment2-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment2-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment3-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment3-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment3-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment3-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment3-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment4-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment4-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment4-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment4-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment4-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment5-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment5-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment5-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment5-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment5-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment6-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment6-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment6-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment6-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment6-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment7-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment7-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment7-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment7-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment7-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment8-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment8-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment8-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment8-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment8-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment9-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment9-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment9-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment9-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment9-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment10-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment10-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment10-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment10-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment10-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment11-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment11-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment11-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment11-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment11-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    
                    # {"experiment_name": "experiment12-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment12-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment12-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    # {"experiment_name": "experiment12-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                     {"experiment_name": "experiment12-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment13-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}}
                ],
        'success': [
                    {"experiment_name": "experiment1-1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-5", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment2-1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-5", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment3-1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-5", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment4-1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-5", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}}
                ]
    }

    if experiment_type in experiments:
        for experiment in experiments[experiment_type]:
            experiment_name = experiment["experiment_name"]
            args_seq = experiment["args_seq"]
            args_model = experiment["args_model"]

            if experiment_type == 'truncation':
                plot_folder = f"plots/{experiment_name}/"
                run_plot_experiment_truncation(experiment_name, data_path, args_seq, args_model, plot_folder)
            else:
                run_experiment(experiment_name, data_path, args_seq, args_model, output_path)

if __name__ == "__main__":
    main()
