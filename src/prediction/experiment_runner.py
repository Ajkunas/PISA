import json
import os
import sys
import pandas as pd
import warnings

from model_trainer import ModelTrainer
from sequence_creator import SequenceCreator
from utils import plot_truncate

warnings.filterwarnings("ignore")

# Run the experiment to predict success by truncating the sequences
def run_plot_experiment_truncation(experiment_name, data_path, args_seq, args_model, plot_folder):
    # Load data
    data = pd.read_csv(data_path)
    
    sequence_creator = SequenceCreator(data=data, **args_seq)
    model_trainer = ModelTrainer(sequence_creator=sequence_creator, **args_model)
    baseline_results = model_trainer.cross_validate()
    
    sequence_methods = ["truncate", "truncate_subsequences"]
    modes = ["first", "last"]
    method_params = [1, 2, 3, 4, 5, 6, 7, 8]
    results = {}
    
    for sequence_method in sequence_methods:
        mean_aucs = {}
        std_aucs = {} 
        for mode in modes: 
            mean_auc_mode = []
            std_auc_mode = []
            for n in method_params:
                args_seq["sequence_method"] = sequence_method
                args_seq["method_mode"] = mode
                args_seq["method_param"] = n
                
                sequence_creator = SequenceCreator(data=data, **args_seq)
                model_trainer = ModelTrainer(sequence_creator=sequence_creator, **args_model)
                results = model_trainer.cross_validate()
                mean_auc_mode.append(results[0]["Mean AUC"])
                std_auc_mode.append(results[0]["Standard Deviation of AUC"])
            
            if sequence_method not in mean_aucs:
                mean_aucs[sequence_method] = {}
            if sequence_method not in std_aucs:
                std_aucs[sequence_method] = {}
            
            mean_aucs[sequence_method][mode] = mean_auc_mode
            std_aucs[sequence_method][mode] = std_auc_mode
        
        # Plot the results
        plot_truncate(sequence_method, method_params, mean_aucs, std_aucs, baseline_results[0]["Mean AUC"], plot_folder)
        

def run_experiment(experiment_name, data_path, args_seq, args_model, output_file):
    # Load data
    data = pd.read_csv(data_path)
    
    # create sequences
    sequence_creator = SequenceCreator(data=data, **args_seq)
    
    model_trainer = ModelTrainer(sequence_creator=sequence_creator, **args_model)

    # Cross validate
    results = model_trainer.cross_validate()

    args_seq['data'] = data_path.split("/")[-1]
    
    # Read the existing content
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Add the new results with experiment_name as the key
    existing_data[experiment_name] = {"args_seq": args_seq, "args_model": args_model, "results": results}


    # Write the updated dictionary back to the file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for {experiment_name} successfully appended to {output_file}")
    
    
if __name__ == "__main__":
    experiment_type = sys.argv[1]
    data_path = sys.argv[2]
    output_file = sys.argv[3]
    
    experiments = []
    output_path = "results/" + output_file # change this to the path where you want to save the results
    
    # Run the experiments
    if experiment_type == 'truncation':
        experiments = [
            {"experiment_name": "experiment1", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
            {"experiment_name": "experiment2", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
            {"experiment_name": "experiment3", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_error", "evaluate": True, "k": 5}},
            {"experiment_name": "experiment4", "args_seq": {"prediction": "success", "activities": [1, 1, 1], "split_type": "median", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "world_code_case", "evaluate": True, "k": 5}},
            ]
        
        for experiment in experiments: 
            experiment_name = experiment["experiment_name"]
            args_seq = experiment["args_seq"]
            args_model = experiment["args_model"]
            plot_folder = "plots/" + experiment_name + "/"
            
            run_plot_experiment_truncation(experiment_name, data_path, args_seq, args_model, plot_folder)
        
    else:
        
        if experiment_type == 'correlation':
    
            experiments = [
                    {"experiment_name": "experiment1-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment1-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "baseline", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment2-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment2-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment3-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment3-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment4-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment4-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_prev", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment5-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment5-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment5-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment5-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment5-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_only", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment6-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment6-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment6-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment6-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment6-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_only", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment7-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment7-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment7-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment7-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment7-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "error_n_only", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment8-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment8-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment8-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment8-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment8-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment9-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment9-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment9-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment9-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment9-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_n_until_error_n", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment10-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment10-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment10-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment10-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment10-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_until", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment11-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment11-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment11-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment11-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment11-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_error_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment12-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment12-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment12-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment12-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment12-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_until", "evaluate": True, "k": 5}},
                    
                    {"experiment_name": "experiment13-1", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-2", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-3", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "rf", "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-4", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": False, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}},
                    {"experiment_name": "experiment13-5", "args_seq": {"prediction": "correlation", "activities": [1, 1, 1], "split_type": "distribution", "one_hot": True, "sequence_method": None, "method_param": 5, "method_mode": 'first'}, "args_model": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "test_size": 0.2, "task": "code_case_n_world_n_prev_only", "evaluate": True, "k": 5}}
                ]

            
        elif experiment_type == 'prediction':
            experiments = [
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
            
        for experiment in experiments:
            experiment_name = experiment["experiment_name"]
            args_seq = experiment["args_seq"]
            args_model = experiment["args_model"]
            run_experiment(experiment_name, data_path, args_seq, args_model, output_path)
            
       # print(f"Finished running {experiment_name}")
