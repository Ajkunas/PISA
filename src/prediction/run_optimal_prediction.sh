#!/bin/bash

data_paths=("/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm.csv" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_123.csv" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_only_l1_l2.csv")
output_paths=("results_optimal_prediction_1.json" "results_optimal_prediction_2.json" "results_optimal_prediction.json")

# Loop over the data paths and run the Python script with each set of arguments
for i in "${!data_paths[@]}"; do
    python run_optimal_prediction.py --data_path "${data_paths[$i]}" --output_path "${output_paths[$i]}"
done
