#!/bin/bash

# Create a temporary Python script to handle the complex command-line argument construction
cat << 'EOF' > run_experiments2_success.py
import os
import json
import subprocess

experiments = [
    {"experiment_name": "experiment1-1", "args": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "split_type": "distribution", "task": "error", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment1-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment1-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "error", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment2-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "case", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment2-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "distribution", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-3", "args": {"model_type": "rf", "split_type": "distribution", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment2-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "distribution", "task": "case", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment3-1", "args": {"model_type": "logistic", "epochs": 200, "lr": 0.01, "dropout": 0, "weight_decay": 0, "split_type": "median", "task": "error", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment3-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "median", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-3", "args": {"model_type": "rf", "split_type": "median", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "median", "task": "error", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment3-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "median", "task": "error", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment4-1", "args": {"model_type": "logistic", "epochs": 100, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "median", "task": "case", "prediction": "success", "one_hot": True, "k": 5}},
    {"experiment_name": "experiment4-2", "args": {"model_type": "logistic", "epochs": 300, "lr": 0.01, "dropout": 0, "weight_decay": 0.01, "split_type": "median", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-3", "args": {"model_type": "rf", "split_type": "median", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-4", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "median", "task": "case", "prediction": "success", "one_hot": False, "k": 5}},
    {"experiment_name": "experiment4-5", "args": {"model_type": "lstm", "epochs": 150, "lr": 0.05, "dropout": 0.1, "weight_decay": 0, "hidden_dim": 100, "split_type": "median", "task": "case", "prediction": "success", "one_hot": True, "k": 5}}
    ]

output_file = "results_experiments2_success.json"
data_path = "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_123.csv"

for experiment in experiments:
    experiment_name = experiment["experiment_name"]
    args = json.dumps(experiment["args"])
    command = f'python experiment_runner.py "{experiment_name}" "{data_path}" "{output_file}" \'{args}\''
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True, check=True)
EOF

python run_experiments2_success.py

rm run_experiments2_success.py
