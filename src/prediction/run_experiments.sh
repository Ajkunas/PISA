#!/bin/bash

# correlation experiments
python experiment_runner.py "correlation" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm.csv" "results_experiment_correlation-1.json"
#python experiment_runner.py "correlation" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_123.csv" "results_experiment_correlation-2.json"
#python experiment_runner.py "correlation" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_only_l1_l2.csv" "results_experiment_correlation-3.json"

# success experiments
python experiment_runner.py "success" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm.csv" "results_experiment_success-1.json"
#python experiment_runner.py "success" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_123.csv" "results_experiment_success-2.json" 
#python experiment_runner.py "success" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm_only_l1_l2.csv" "results_experiment_success-3.json"

# truncation methods 
python experiment_runner.py "truncation" "/Users/ajkunaseipi/Documents/MA4/PISA/data/robotarm.csv" "results_experiment_trunc-1.json"

# Add other experiments if needed