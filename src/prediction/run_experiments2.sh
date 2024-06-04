#!/bin/bash

# Create a temporary Python script to handle the complex command-line argument construction
cat << 'EOF' > run_experiments2.py
import os
import json
import subprocess

experiments = [
    {"experiment_name": "experiment1-1", "args": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "split_type": "distribution", "task": "baseline", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment1-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "baseline", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "baseline", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "baseline", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "baseline", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment2-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_world_n_prev", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment2-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_world_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_error_n_world_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_world_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_world_n_prev", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment3-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment3-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.2, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment4-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_prev", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment4-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_error_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_prev", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_prev", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment5-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment5-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment5-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment5-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment5-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment6-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment6-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment6-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment6-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment6-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment7-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "error_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment7-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment7-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment7-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "error_n_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment7-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "error_n_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment8-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_until", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment8-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment8-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment8-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment8-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_until", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment9-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_until_error_n", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment9-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_n_until_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment9-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_n_until_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment9-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_until_error_n", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment9-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_n_until_error_n", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment10-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_case_n_until", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment10-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_case_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment10-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_case_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment10-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_case_n_until", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment10-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_case_n_until", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment11-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_world_n_prev_only", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment11-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "code_error_n_world_n_prev_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment11-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "code_error_n_world_n_prev_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment11-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_world_n_prev_only", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment11-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "code_error_n_world_n_prev_only", "one_hot": True, "k": 5}}
]

output_file = "results_experiment2_update.json"
data_path = "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_123.csv"

for experiment in experiments:
    experiment_name = experiment["experiment_name"]
    args = json.dumps(experiment["args"])
    command = f'python experiment_runner.py "{experiment_name}" "{data_path}" "{output_file}" \'{args}\''
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True, check=True)
EOF

python run_experiments2.py

rm run_experiments2.py
