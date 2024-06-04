import pandas as pd 

from preprocess import *
from plots_analysis import *

import warnings
warnings.filterwarnings('ignore')



def run_preprocessing():
    
    # Paths to the data files, change accordingly
    folder_path = "../data/"
    world_data = "ldw_2023_pilot_coding_tasks_outputs_processed_3.csv"
    code_data = "data_code_space.csv"
    
    df = pd.read_csv(folder_path + world_data)
    df = df[df['Activity ID'].isin(['P1M120', 'P1M123', 'P1M124', 'P1M128'])]

    df_code = pd.read_csv(folder_path + code_data)
    df_code = df_code.drop(columns=['Unnamed: 0'])
    df_code = df_code.dropna(subset=['XML'])

    robotarm_df = df.merge(df_code, on=['Student ID', 'Activity ID', 'timestamp'])
    
    KEY_VECTORS = [KEY_VECTOR_L1, KEY_VECTOR_L2, KEY_VECTOR_L3]
    INIT_WORLDSPACES = [INIT_WORLDSPACE_L1, INIT_WORLDSPACE_L2, INIT_WORLDSPACE_L3]
    ACTIVITY_IDS = [L1, L2, L3]
    ACTIVITY_NBS = [1, 2, 3]

    # Defaut penalties for the robot arm
    print("Preprocessing robot arm data with default penalties...")
    robotarm = preprocessing_general(robotarm_df, KEY_VECTORS, INIT_WORLDSPACES, ACTIVITY_IDS, ACTIVITY_NBS, PENALTIES, code=True)

    robotarm_l1 = robotarm[robotarm['activity'] == 1]
    robotarm_l2 = robotarm[robotarm['activity'] == 2]
    robotarm_l3 = robotarm[robotarm['activity'] == 3]
    
    print("Saving files...")
    robotarm.to_csv(folder_path + 'robotarm.csv', index=False)
    robotarm_l1.to_csv(folder_path + 'robotarm_l1.csv', index=False)
    robotarm_l2.to_csv(folder_path + 'robotarm_l2.csv', index=False)
    robotarm_l3.to_csv(folder_path + 'robotarm_l3.csv', index=False)
    
    print("Preprocessing robot arm data with no penalties and optimal penalties...")
    robotarm_no_penalties = preprocessing_general(robotarm_df, KEY_VECTORS, INIT_WORLDSPACES, ACTIVITY_IDS, ACTIVITY_NBS, NO_PENALTIES, code=True)
    robotarm_no_penalties.to_csv(folder_path + 'robotarm_no_penalties.csv', index=False)
    
    robotarm_optimal_penalties = preprocessing_general(robotarm_df, KEY_VECTORS, INIT_WORLDSPACES, ACTIVITY_IDS, ACTIVITY_NBS, OPTIMAL_PENALTIES, code=True)
    robotarm_optimal_penalties.to_csv(folder_path + 'robotarm_optimal_penalties.csv', index=False)
    
    print("Saving files robotarm_only_l1_l2.csv...")
    # useful for prediction 
    robotarm_only_l1_l2 = pd.concat([robotarm_l1, robotarm_l2])
    robotarm_only_l1_l2.to_csv(folder_path + 'robotarm_only_l1_l2.csv', index=False)
    
    print("Saving files robotarm_123.csv...")
    robotarm_123 = robotarm[robotarm['activity_all'] == "123"]
    robotarm_123.to_csv(folder_path + 'robotarm_123.csv', index=False)
    
    
if __name__ == "__main__":
    
    run_preprocessing()
    
    print("Finished preprocessing and file saved.")