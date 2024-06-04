import pandas as pd 
import numpy as np

from plots_pattern import *

import warnings
warnings.filterwarnings('ignore')


def generate_heatmap_plot(): 
    
    folder = "plots/" # Folder to save the plots
    date = "2024_06_04_0" # Date of the experiment

    # General patterns
    path1 = f'experiments/group_cases/{date}/results.pkl'
    experiment1 = "Case sequences"

    path2 = f'experiments/group_error/{date}/results.pkl'
    experiment2 = "Error sequences"

    path3 = f'experiments/group_world_dis/{date}/results.pkl'
    experiment3 = "World distances with distribtuion split"

    path4 = f'experiments/group_code_dis/{date}/results.pkl'
    experiment4 = "Code distances with distribtuion split"

    path5 = f'experiments/group_world_med/{date}/results.pkl'
    experiment5 = "World distances with median split"

    path6 = f'experiments/group_code_med/{date}/results.pkl'
    experiment6 = "Code distances with median split"

    paths = [path1, path2, path3, path4, path5, path6]
    experiments = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]

    for path, experiment in zip(paths, experiments):
        generate_plots(path, experiment, folder)
        
    print("Heatmap plots generated successfully")
    
    # First attempts pattern mining
    path1 = f'experiments/group_first_ca/{date}/results.pkl'
    experiment1 = "Case sequences for first attempts"

    path2 = f'experiments/group_first_er/{date}/results.pkl'
    experiment2 = "Error sequences for first attempts"

    path3 = f'experiments/group_first_wo/{date}/results.pkl'
    experiment3 = "World distances with distribtuion split for first attempts"

    path4 = f'experiments/group_first_cod/{date}/results.pkl'
    experiment4 = "Code distances with distribtuion split for first attempts"

    paths = [path1, path2, path3, path4]
    experiments = [experiment1, experiment2, experiment3, experiment4]

    for path, experiment in zip(paths, experiments):
        generate_plots(path, experiment, folder)
        
    print("Heatmap plots for first attempts generated successfully")
        
        
    # Last attempts pattern mining
    path1 = f'experiments/group_last_ca/{date}/results.pkl'
    experiment1 = "Case sequences for last attempts"

    path2 = f'experiments/group_last_er/{date}/results.pkl'
    experiment2 = "Error sequences for last attempts"

    path3 = f'experiments/group_last_wo/{date}/results.pkl'
    experiment3 = "World distances with distribtuion split for last attempts"

    path4 = f'experiments/group_last_cod/{date}/results.pkl'
    experiment4 = "Code distances with distribtuion split for last attempts"

    paths = [path1, path2, path3, path4]
    experiments = [experiment1, experiment2, experiment3, experiment4]

    for path, experiment in zip(paths, experiments):
        generate_plots(path, experiment, folder)
        
    print("Heatmap plots for last attempts generated successfully")
    
    
# main function
if __name__ == "__main__":
    
    generate_heatmap_plot()
    