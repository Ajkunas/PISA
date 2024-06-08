import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pickle


def plot_heatmap_presence(data, test, experiment, folder):
    
    dfs = []
    dfs_i_support = []
    groups = []
    
    for group in data[test].keys():
        df = data[test][group]['patterns'].dropna()
        df['S-Support (S-Frequency %)'] = df['S-Support (S-Frequency %)'].astype(float)
        df['I-Support (I-Frequency Mean)'] = df['I-Support (I-Frequency Mean)'].astype(float)
        df_s_support = df[['Pattern', 'S-Support (S-Frequency %)']]
        df_i_support = df[['Pattern', 'I-Support (I-Frequency Mean)']]
        dfs.append(df_s_support)
        dfs_i_support.append(df_i_support)
        groups.append(group)
    
    # Merge the two data frames on the 'Pattern' column
    suffixes = [f'_{i}' for i in groups]
    if len(dfs) == 0:
        print(f"No data available for {test} in {experiment}")
        return
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Pattern', how='outer', suffixes=('', suffixes.pop(1))), dfs)
    
    suffixes_i_support = [f'_{i}' for i in groups]
    merged_df_i_support = reduce(lambda left, right: pd.merge(left, right, on='Pattern', how='outer', suffixes=('', suffixes_i_support.pop(1))), dfs_i_support)

    # Set the 'Pattern' column as the index
    merged_df.set_index('Pattern', inplace=True)
    merged_df_i_support.set_index('Pattern', inplace=True)

    # Fill NaN values with 0
    merged_df.fillna(0, inplace=True)
    merged_df_i_support.fillna(0, inplace=True)

    presence_matrix = merged_df_i_support.values

    # Plot the histogram
    plt.figure(figsize=(12, 8))
    plt.imshow(merged_df.values, cmap='magma', aspect='auto')

    # Add text annotations for each cell
    for i in range(presence_matrix.shape[0]):
        for j in range(presence_matrix.shape[1]):
            plt.text(j, i, '{:.2f}'.format(presence_matrix[i, j]), ha='center', va='center', color='black')

    plt.xticks(range(len(groups)), groups)
    plt.xticks(rotation=45)
    plt.yticks(np.arange(len(merged_df)), merged_df.index)
    plt.ylabel('Pattern')
    plt.colorbar(label='S-Support')
    plt.title('Pattern Presence for Experiment: ' + experiment)
    plt.tight_layout()
    plt.savefig(f'./{folder}/{experiment}_{test}.png')
    #plt.show()
    
    
def generate_plots(path, experiment, folder):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    for test in data.keys(): 
        if test != 'general':
            plot_heatmap_presence(data, test, experiment, folder)