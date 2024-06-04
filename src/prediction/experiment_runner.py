import json
import os
import sys
import pandas as pd
import warnings
from train import cross_validate 

warnings.filterwarnings("ignore")

def run_experiment(experiment_name, data_path, args, output_file):
    # Load data
    data = pd.read_csv(data_path)
    results = cross_validate(**args, data=data)
    
    args['data'] = data_path.split("/")[-1]
    
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
    existing_data[experiment_name] = {"args": args, "results": results}

    # Write the updated dictionary back to the file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for {experiment_name} successfully appended to {output_file}")

    
    
if __name__ == "__main__":
    experiment_name = sys.argv[1]
    data_path = sys.argv[2]
    output_file = sys.argv[3]
    args = json.loads(sys.argv[4])
    
    run_experiment(experiment_name, data_path, args, output_file)
