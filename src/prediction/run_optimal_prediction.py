import json
import numpy as np
import pandas as pd
import argparse

from train import cross_validate
from train_optimal import cross_validate_optimized

import warnings
warnings.filterwarnings("ignore")



# Function to compare results and find the closest match
def find_closest_match(baseline_results, method_results, metric="Mean AUC"):
    baseline_metric = float(baseline_results[0][metric])
    closest_n = None
    closest_diff = float('inf')
    optimal_diff = float(baseline_results[0]["Standard Deviation of AUC"])
    
    for n, results in method_results.items():
        method_metric = float(results[0][metric])
        diff = abs(baseline_metric - method_metric)
        if diff < optimal_diff:
            closest_n = n
            return closest_n, closest_diff
        else:
            if diff < closest_diff:
                closest_n = n
                closest_diff = diff
    
    return closest_n, closest_diff


def compare_results(args, method_mode):
    
    baseline_results = cross_validate(**args)
    print("Baseline Results:", baseline_results)

    # Store method results
    method_params = [2, 3, 4, 5, 6, 7, 8]
    method_results = {} 
    
    args_trunc = args.copy()
    args_trunc["sequence_method"] = "truncate"
    args_trunc["method_mode"] = method_mode
    args_trunc = {k: v for k, v in args_trunc.items() if k != "prediction"}
    
    for n in method_params:
        results = cross_validate_optimized(**args_trunc, method_param=n)
        method_results[n] = results
        
    # Find the closest match
    closest_n, closest_diff = find_closest_match(baseline_results, method_results)
    print(f"The parameter n that gives results closest to the baseline is {closest_n} with a difference of {closest_diff:.2f}")
    
    args_to_save = {k: v for k, v in args.items() if k != "data"}
    
    results = {"args": args_to_save, "baseline_results": baseline_results, "closest_n": closest_n, "closest_diff": closest_diff}
    
    return results


def run_optimal_prediction(data_path, output_path):
    # Load data
    df = pd.read_csv(data_path) # Load your dataframe here
    
    results_to_save = []

    # Best results found
    args1 = {
        "model_type": "rf",
        "split_type": "median",
        "task": "error",
        "prediction": "success",
        "one_hot": False,
        "k": 5,
        "data": df
    }
    
    results1 = compare_results(args1, "last")
    results_to_save.append(results1)
    
    args2 = {
        "model_type": "rf",
        "split_type": "median",
        "task": "case",
        "prediction": "success",
        "one_hot": False,
        "k": 5,
        "data": df
    }

    results2 = compare_results(args2, "last")
    results_to_save.append(results2)
    
    args3 = {
        "model_type": "rf",
        "split_type": "distribution",
        "task": "error",
        "prediction": "success",
        "one_hot": False,
        "k": 5,
        "data": df
    }

    results3 = compare_results(args3, "last")
    results_to_save.append(results3)
    
    args4 = {
        "model_type": "rf",
        "split_type": "distribution",
        "task": "case",
        "prediction": "success",
        "one_hot": False,
        "k": 5,
        "data": df
    }

    results4 = compare_results(args4, "last")
    results_to_save.append(results4)
    
    with open(output_path, "w") as f:
        json.dump(results_to_save, f)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimal prediction script.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    run_optimal_prediction(args.data_path, args.output_path)