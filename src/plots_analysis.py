import pandas as pd 
from distances import preprocess, euclidean_v2
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from scipy import stats


def plots_per_student(df, filter, folder): 
    
    df_success = df[df['success'] == 1] 
    df_fail = df[df['success'] == 0]
    
    nb_student_total = len(df['Student ID'].unique())
    nb_student_success = len(df_success['Student ID'].unique())
    nb_student_fail = len(df_fail['Student ID'].unique())
    
    df_filtered = df[df['nb_tentative'] <= 10]
    df_success_filtered = df_success[df_success['nb_tentative'] <= 10]
    df_failure_filtered = df_fail[df_fail['nb_tentative'] <= 10]
    
    nb_total_filtered = len(df_filtered['Student ID'].unique())
    nb_success_filtered = len(df_success_filtered['Student ID'].unique())
    nb_fail_filtered = len(df_failure_filtered['Student ID'].unique())
    
    # plot distribution of tentatives
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['nb_tentative'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_success['nb_tentative'], bins=20, alpha=0.5, label='Success')
    ax.hist(df_fail['nb_tentative'], bins=20, alpha=0.5, label='Failure')
    
    ax.set_xlabel('Number of tentatives')
    ax.set_ylabel('Number of students')
    ax.legend()
    ax.set_title('Distribution of tentatives')
    plt.savefig(f"../plot/{folder}/tentatives_per_student_hist.png")
    plt.show()
    
    # plot boxplot of tentatives
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=filter, y='nb_tentative', data=df, ax=ax)
    ax.set_ylabel('Number of tentatives')
    ax.set_title('Boxplot of tentatives')
    plt.savefig(f"../plot/{folder}/tentatives_per_student_boxplot.png")
    plt.show()
    
    # distribution of euclidean distance
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['mean_euclidean_distance'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_success['mean_euclidean_distance'], bins=20, alpha=0.5, label='Success')
    ax.hist(df_fail['mean_euclidean_distance'], bins=20, alpha=0.5, label='Failure')
    
    ax.set_xlabel('Mean euclidean distance')
    ax.set_ylabel('Number of students')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title('Distribution of euclidean distance')
    plt.savefig(f"../plot/{folder}/euclidean_distance_per_student_hist.png")
    plt.show()
    
    # plot boxplot of euclidean distance
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.boxplot(x=filter, y='mean_euclidean_distance', data=df, ax=ax)
    ax.set_ylabel('Mean euclidean distance')
    ax.set_title('Boxplot of euclidean distance')
    plt.savefig(f"../plot/{folder}/euclidean_distance_per_student_boxplot.png")
    plt.show()
    
    # plot euclidean distance per nb_tentative
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='nb_tentative', y='mean_euclidean_distance', hue='success', data=df_filtered,
                 err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
    ax.set_ylabel('Mean Euclidean distance for each Total Tentative')
    ax.set_xlabel("Nb of Total Tentative")
    ax.set_title('Euclidean distance')
    plt.savefig(f"../plot/{folder}/euclidean_distance_nb_tentative.png")
    
    # distribution of delta successive
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['mean_delta_successive'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_success['mean_delta_successive'], bins=20, alpha=0.5, label='Success')
    ax.hist(df_fail['mean_delta_successive'], bins=20, alpha=0.5, label='Failure')
    
    ax.set_xlabel('Mean delta successive')
    ax.set_ylabel('Number of students')
    ax.legend()
    ax.set_title('Distribution of delta successive')
    plt.savefig(f"../plot/{folder}/delta_successive_per_student_hist.png")
    plt.show()
    
    # plot boxplot of delta successive
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.boxplot(x=filter, y='mean_delta_successive', data=df, ax=ax)
    ax.set_ylabel('Mean delta successive')
    ax.set_title('Boxplot of delta successive')
    plt.savefig(f"../plot/{folder}/delta_successive_per_student_boxplot.png")
    plt.show()
    
    # plot delta successive per nb_tentative
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='nb_tentative', y='mean_delta_successive', hue='success', data=df_filtered, ax=ax, 
                 err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
    ax.set_ylabel('Mean Successive Euclidean distance for each Total Tentative')
    ax.set_xlabel("Nb of Total Tentative")
    ax.set_title('Euclidean distance')
    plt.savefig(f"../plot/{folder}/delta_successive_nb_tentative.png")
    
    
def plots(df, filter, folder): 
    
    tot_tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    df_true = df[df[filter] == 1]
    df_false = df[df[filter] == 0]
    
    nb_total = len(df['Student ID'].unique())
    nb_student_true = len(df_true['Student ID'].unique())
    nb_student_false = len(df_false['Student ID'].unique())
    
    df_filtered = df[df['nb_tentative'] <= 10]
    df_true_filtered = df_true[df_true['nb_tentative'] <= 10]
    df_false_filtered = df_false[df_false['nb_tentative'] <= 10]
    
    nb_total_filtered = len(df_filtered['Student ID'].unique())
    nb_true_filtered = len(df_true_filtered['Student ID'].unique())
    nb_false_filtered = len(df_false_filtered['Student ID'].unique())
    
    # plot euclidean distance
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.boxplot(x=filter, y='euclidean_distance', hue=filter, data=df, ax=ax)
    ax.set_ylabel('Euclidean distance')
    ax.set_title('Boxplot of euclidean distance')
    plt.savefig(f"../plot/{folder}/euclidean_distance_boxplot.png")
    plt.show()
    
    # plot euclidean distance histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['euclidean_distance'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_true['euclidean_distance'], bins=20, alpha=0.5, label='True')
    ax.hist(df_false['euclidean_distance'], bins=20, alpha=0.5, label='False')
    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('Number of students')
    ax.set_title('Distribution of euclidean distance')
    plt.savefig(f"../plot/{folder}/euclidean_distance_hist.png")
    plt.show()
    
    # plot line plot of euclidean distance per pct_activity
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='pct_activity', y='euclidean_distance', hue=filter,  data=df_filtered, 
                 err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
    ax.set_ylabel('Euclidean distance')
    ax.set_xlabel("Percentage of activity")
    ax.set_title(f'Evolution of euclidean distance, True: {nb_true_filtered}, False: {nb_false_filtered}')
    plt.savefig(f"../plot/{folder}/euclidean_distance_pct_activity.png")
    plt.show()
    
    # Multi plot of euclidean distance per tentative 
    fig, axs = plt.subplots(4, 3, figsize=(15, 12))

    for i, ax in enumerate(axs.flat):
        if i < len(tot_tentatives):
            grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
            
            # count number of failing and successful students
            nb_false_student = len(df_false[df_false['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())
            nb_true_student = len(df_true[df_true['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())

            sns.lineplot(data=grouped_data, y=f'euclidean_distance', x='index', hue=filter,
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
            
            ax.set_title(f"Total Tentatives: {tot_tentatives[i]}")
            ax.set_xlabel("Tentative")
            ax.set_ylabel("Euclidean distance")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                if label == '0':
                    new_labels.append(f"{filter} : {label}, N={nb_false_student}")
                else:
                    new_labels.append(f"{filter} : {label}, N={nb_true_student}")
            ax.legend(handles, new_labels, loc='upper right', fontsize=9)
            
        else:
            ax.axis('off')

    plt.suptitle("Euclidean distance")
    plt.tight_layout()
    plt.savefig(f"../plot/{folder}/euclidean_distance_per_tentative_multi.png")
    plt.show()
    
    # plot delta successive
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.boxplot(x=filter, y='delta_successive', hue=filter, data=df, ax=ax)
    ax.set_ylabel('Successive Euclidean distance')
    ax.set_title('Boxplot of successive euclidean distance')
    plt.savefig(f"../plot/{folder}/delta_successive_boxplot.png")
    plt.show()
    
    # plot delta successive histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['delta_successive'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_true['delta_successive'], bins=20, alpha=0.5, label='True')
    ax.hist(df_false['delta_successive'], bins=20, alpha=0.5, label='False')
    ax.set_xlabel('Successive Euclidean distance')
    ax.set_ylabel('Number of students')
    ax.set_title('Distribution of successive euclidean distance')
    plt.savefig(f"../plot/{folder}/delta_successive_hist.png")
    plt.show()
    
    # plot line plot of delta_successive per pct_activity
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='pct_activity', y='delta_successive', hue=filter,  data=df_filtered, 
                 err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
    ax.set_ylabel('Successive Euclidean distance')
    ax.set_xlabel("Percentage of activity")
    ax.set_title(f'Evolution of successive euclidean distance, True: {nb_true_filtered}, False: {nb_false_filtered}')
    plt.savefig(f"../plot/{folder}/delta_successive_pct_activity.png")
    plt.show()
    
    # Multi plot of delta_successive per tentative 
    fig, axs = plt.subplots(4, 3, figsize=(15, 12))

    for i, ax in enumerate(axs.flat):
        if i < len(tot_tentatives):
            grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
            
            # count number of failing and successful students
            nb_false_student = len(df_false[df_false['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())
            nb_true_student = len(df_true[df_true['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())

            sns.lineplot(data=grouped_data, y=f'delta_successive', x='index', hue=filter,
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
            
            ax.set_title(f"Total Tentatives: {tot_tentatives[i]}")
            ax.set_xlabel("Tentative")
            ax.set_ylabel("Successive euclidean distance")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                if label == '0':
                    new_labels.append(f"False, N={nb_false_student}")
                else:
                    new_labels.append(f"True, N={nb_true_student}")
            ax.legend(handles, new_labels, loc='upper right', fontsize=9)
            
        else:
            ax.axis('off')

    plt.suptitle("Euclidean distance")
    plt.tight_layout()
    plt.savefig(f"../plot/{folder}/delta_successive_per_tentative_multi.png")
    plt.show()
    

def plots_errors_per_student(df, filter, folder): 
    
    tot_tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    df_success = df[df['success'] == 1] 
    df_fail = df[df['success'] == 0]
    
    df_filtered = df[df['nb_tentative'] <= 10]
    df_success_filtered = df_success[df_success['nb_tentative'] <= 10]
    df_failure_filtered = df_fail[df_fail['nb_tentative'] <= 10]
    
    # plot distribution of errors
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['pct_error'], bins=20, alpha=0.5, label='Overall')
    ax.hist(df_success['pct_error'], bins=20, alpha=0.5, label='Success')
    ax.hist(df_fail['pct_error'], bins=20, alpha=0.5, label='Failure')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Percentage of errors')
    ax.set_ylabel('Number of students')
    ax.legend()
    ax.set_title('Distribution of errors')
    plt.savefig(f"../plot/{folder}/errors_per_student_hist.png")
    plt.show()
    
    # plot boxplot of errors
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=filter, y='pct_error', data=df, ax=ax)
    ax.set_ylabel('Percentage of errors')
    ax.set_title('Boxplot of errors')
    plt.savefig(f"../plot/{folder}/errors_per_student_boxplot.png")
    plt.show()
    
    # distribution of errors
    for error in ['move1', 'move2', 'pickup1', 'place1']:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(df[f'pct_{error}'], bins=20, alpha=0.5, label='Overall')
        ax.hist(df_success[f'pct_{error}'], bins=20, alpha=0.5, label='Success')
        ax.hist(df_fail[f'pct_{error}'], bins=20, alpha=0.5, label='Failure')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel(f'Percentage of {error}')
        ax.set_ylabel('Number of students')
        ax.legend()
        ax.set_title(f'Distribution of {error}')
        plt.savefig(f"../plot/{folder}/{error}_per_student_hist.png")
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=filter, y=f'pct_{error}', data=df, ax=ax)
        ax.set_ylabel(f'Percentage of {error}')
        ax.set_title(f'Boxplot of {error}')
        plt.savefig(f"../plot/{folder}/{error}_per_student_boxplot.png")
        plt.show()
        
    
    # plot error_rate per nb_tentative
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='nb_tentative', y='pct_error', hue='success', data=df_filtered, ax=ax, 
                 err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
    ax.set_ylabel('Mean Error rate for each Total Tentative')
    ax.set_xlabel("Nb of Total Tentative")
    ax.set_title('Error Rate')
    plt.savefig(f"../plot/{folder}/errors_per_student_nb_tentative.png")
    

def plots_errors(df, filter, folder): 
    
    tot_tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    df_success = df[df['success'] == 1] 
    df_fail = df[df['success'] == 0]
    
    df_filtered = df[df['nb_tentative'] <= 10]
    df_success_filtered = df_success[df_success['nb_tentative'] <= 10]
    df_failure_filtered = df_fail[df_fail['nb_tentative'] <= 10]
    
    for error in ['error', 'move1', 'move2', 'pickup1', 'place1']:
        # Multi plot of euclidean distance per tentative 
        fig, axs = plt.subplots(4, 3, figsize=(15, 12))

        for i, ax in enumerate(axs.flat):
            if i < len(tot_tentatives):
                grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
                
                # count number of failing and successful students
                nb_fail_student = len(df_fail[df_fail['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())
                nb_success_student = len(df_success[df_success['nb_tentative'] == tot_tentatives[i]]['Student ID'].unique())

                sns.lineplot(data=grouped_data, y=error, x='index', hue=filter,
                            err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
                
                ax.set_title(f"Total Tentatives: {tot_tentatives[i]}")
                ax.set_xlabel("Tentative")
                ax.set_ylabel(f"{error} rate")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                handles, labels = ax.get_legend_handles_labels()
                new_labels = []
                for label in labels:
                    if label == '0':
                        new_labels.append(f"Failing, N={nb_fail_student}")
                    else:
                        new_labels.append(f"Successful, N={nb_success_student}")
                ax.legend(handles, new_labels, loc='upper right', fontsize=9)
                
            else:
                ax.axis('off')

        plt.suptitle(f"{error} Rate")
        plt.tight_layout()
        plt.savefig(f"../plot/{folder}/{error}_per_tentative_multi.png")
        plt.show()
        
    for col in ['nb_tentative', 'pct_activity']:
        fig, ax = plt.subplots(figsize=(10, 5))
        for error in ['error','move1', 'move2', 'pickup1', 'place1']:
            sns.lineplot(x=col, y=error, data=df, ax=ax, 
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, label=error, legend=True)
        ax.set_ylabel(f'{error} rate')
        ax.set_xlabel(col)
        ax.set_title(f'Mean Error Rate per {col}')
        ax.legend(loc='upper right')
        plt.savefig(f"../plot/{folder}/errors_type_{col}.png")
        plt.show()
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
        for error in ['move1', 'move2', 'pickup1', 'place1']:
            sns.lineplot(x=col, y=error, data=df_success_filtered, ax=ax[0], 
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, label=error, legend=True)
        ax[0].set_ylabel(f'{error} rate')
        ax[0].set_xlabel(col)
        ax[0].set_title(f'{error} Rate per {col} for Successful Students')
        ax[0].legend(loc='upper right')
        
        for error in ['move1', 'move2', 'pickup1', 'place1']:
            sns.lineplot(x=col, y=error, data=df_failure_filtered, ax=ax[1], 
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, label=error, legend=True)
        
        ax[1].set_ylabel(f'{error} rate')
        ax[1].set_xlabel(col)
        ax[1].set_title(f'Mean {error} Rate per {col} for Failing Students')
        ax[1].legend(loc='upper right')
        
        plt.savefig(f"../plot/{folder}/errors_type_{col}_succ_fail.png")
        plt.show()
    
    
def plots_distance_2dim(df, folder, filter): 
    tot_tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    for col in ['euclidean_distance', 'delta_successive']:
        for i in range(len(tot_tentatives)):
            grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
            
            count_error_per_tentative_success = grouped_data[grouped_data['success'] == 1].groupby('index').agg({'error': 'sum'}).reset_index()
            count_error_per_tentative_fail = grouped_data[grouped_data['success'] == 0].groupby('index').agg({'error': 'sum'}).reset_index()

            count_error_per_tentative_success['success'] = 1
            count_error_per_tentative_fail['success'] = 0

            count_error_per_tentative = pd.concat([count_error_per_tentative_success, count_error_per_tentative_fail])
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            sns.barplot(x='index', y='error', hue='success', data=count_error_per_tentative, ax=axs[0])
            axs[0].set_xlabel('Tentative')
            axs[0].set_ylabel('Number of error')
            axs[0].set_title('Number of Error per Tentative')
            
            nb_success_student = len(grouped_data[grouped_data['success'] == 1]['Student ID'].unique())
            nb_failing_student = len(grouped_data[grouped_data['success'] == 0]['Student ID'].unique())

            sns.lineplot(data=grouped_data, y=col, x='index', hue=filter,
                            err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=axs[1], legend=True)
            
            axs[1].set_xlabel("Tentative")
            axs[1].set_ylabel(col)
            axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
            plt.suptitle(f"Total Tentatives={tot_tentatives[i]}, N={nb_success_student} successful students, N={nb_failing_student} failing students")
            plt.savefig(f"../plot/{folder}/barplot_error_{col}_{tot_tentatives[i]}.png")
            plt.show()
    
    for col in ['euclidean_distance', 'delta_successive']: 
        # plotting the CASE for successful students
        for i in range(len(tot_tentatives)):
            grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
            
            count_error_per_tentative_success = grouped_data[grouped_data['success'] == 1].groupby('index').agg({'case1': 'sum', 'case2': 'sum', 'case3':'sum'}).reset_index()
            count_error_per_tentative_fail = grouped_data[grouped_data['success'] == 0].groupby('index').agg({'case1': 'sum', 'case2': 'sum', 'case3':'sum'}).reset_index()

            count_error_per_tentative_success['success'] = 1
            count_error_per_tentative_fail['success'] = 0

            count_error_per_tentative = pd.concat([count_error_per_tentative_success, count_error_per_tentative_fail])
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            sns.barplot(x='index', y='case1', data=count_error_per_tentative_success, color='blue', label='Case 1', ax=axs[0])
            sns.barplot(x='index', y='case2', data=count_error_per_tentative_success, color='orange', label='Case 2', ax=axs[0])
            sns.barplot(x='index', y='case3', data=count_error_per_tentative_success, color='green', label='Case 3', ax=axs[0])
            
            axs[0].set_xlabel('Tentative')
            axs[0].set_ylabel('Count')
            axs[0].legend()
            axs[0].set_title('Number of Errors per Tentative for Successful Students')
            
            nb_success_student = len(grouped_data[grouped_data['success'] == 1]['Student ID'].unique())
            nb_failing_student = len(grouped_data[grouped_data['success'] == 0]['Student ID'].unique())

            sns.lineplot(data=grouped_data[grouped_data['success'] == 1], y=col, x='index',
                            err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=axs[1], legend=True)
            
            #ax.set_ylim(0, 1)
            #axs[0].set_title(f"Total Tentatives: {tot_tentatives[i]}")
            axs[1].set_xlabel("Tentative")
            axs[1].set_ylabel(col)
            axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
                
            plt.suptitle(f"Total Tentatives={tot_tentatives[i]}, N={nb_success_student} Successful Students")
            plt.savefig(f"../plot/{folder}/barplot_error_cases_{col}_tentative_{tot_tentatives[i]}_success.png")
            plt.show()
            
            # for failing students
            for i in range(len(tot_tentatives)):
                grouped_data = df[df['nb_tentative'] == tot_tentatives[i]]
                
                count_error_per_tentative_success = grouped_data[grouped_data['success'] == 1].groupby('index').agg({'case1': 'sum', 'case2': 'sum', 'case3':'sum'}).reset_index()
                count_error_per_tentative_fail = grouped_data[grouped_data['success'] == 0].groupby('index').agg({'case1': 'sum', 'case2': 'sum', 'case3':'sum'}).reset_index()

                count_error_per_tentative_success['success'] = 1
                count_error_per_tentative_fail['success'] = 0

                count_error_per_tentative = pd.concat([count_error_per_tentative_success, count_error_per_tentative_fail])
                
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                
                sns.barplot(x='index', y='case1', data=count_error_per_tentative_fail, color='blue', label='Case 1', ax=axs[0])
                sns.barplot(x='index', y='case2', data=count_error_per_tentative_fail, color='orange', label='Case 2', ax=axs[0])
                sns.barplot(x='index', y='case3', data=count_error_per_tentative_fail, color='green', label='Case 3', ax=axs[0])
                
                axs[0].set_xlabel('Tentative')
                axs[0].set_ylabel('Count')
                axs[0].legend()
                axs[0].set_title('Number of Errors per Tentative for Failing Students')
                
                nb_success_student = len(grouped_data[grouped_data['success'] == 1]['Student ID'].unique())
                nb_failing_student = len(grouped_data[grouped_data['success'] == 0]['Student ID'].unique())

                sns.lineplot(data=grouped_data[grouped_data['success'] == 0], y=col, x='index',
                                err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=axs[1], legend=True)
                
                #ax.set_ylim(0, 1)
                #axs[0].set_title(f"Total Tentatives: {tot_tentatives[i]}")
                axs[1].set_xlabel("Tentative")
                axs[1].set_ylabel(col)
                axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                    
                plt.suptitle(f"Total Tentatives={tot_tentatives[i]}, N={nb_failing_student} Failing Students")
                plt.savefig(f"../plot/{folder}/barplot_error_cases_{col}_tentative_{tot_tentatives[i]}_fail.png")
                plt.show()
    
    
def plots_comparision_tasks(df, folder, filter_task): 
    
    df_grouped_per_student_per_task = df.groupby(['Student ID', filter_task]).agg({'success': 'max', 
                                                                                   'nb_tentative': 'max', 
                                                                                   'euclidean_distance': 'mean', 
                                                                                   'delta_successive': 'mean', 
                                                                                   'error': 'mean'}).reset_index()
    df_grouped_per_student_per_task.columns = ['Student ID', filter_task, 'success', 
                                               'nb_tentative', 'mean_euclidean_distance', 
                                               'mean_delta_successive', 'pct_error']
    
    df_filtered = df[df['nb_tentative'] <= 10]
    
    tasks = df[filter_task].unique()
    
    # bar plot of number of students per task
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=filter_task, data=df_grouped_per_student_per_task, ax=ax)
    ax.set_ylabel('Number of students')
    ax.set_title('Number of students per task')
    plt.savefig(f"../plot/{folder}/students_per_{filter_task}_barplot.png")
    plt.show()
    
    # for each task, count the number of successful and failing students
    for task in tasks:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        sns.countplot(x='success', data=df_grouped_per_student_per_task[df_grouped_per_student_per_task[filter_task] == task], ax=ax)
        ax.set_ylabel('Number of students')
        ax.set_title(f'Number of students for task {task}')
        plt.savefig(f"../plot/{folder}/students_per_{filter_task}_{task}_barplot.png")
        plt.show()
    
        for col in ['euclidean_distance', 'delta_successive']:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            sns.boxplot(x='success', y=col, data=df[df[filter_task] == task], ax=ax)
            ax.set_ylabel(col)
            ax.set_title(f'Boxplot of {col} for task {task}')
            plt.savefig(f"../plot/{folder}/{col}_{filter_task}_{task}_boxplot.png")
            plt.show()
    
        # line plot of euclidean distance per task
        for col in ['nb_tentative', 'pct_activity']:
            for type in ['euclidean_distance', 'delta_successive', 'error']:
                fig, ax = plt.subplots(figsize=(10, 5))
                if col == 'nb_tentative':
                    sns.lineplot(x=col, y=type, hue='success', data=df[df[filter_task] == task], ax=ax,
                                    err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
                else: 
                    sns.lineplot(x=col, y=type, hue='success', data=df_filtered[df_filtered[filter_task] == task], ax=ax,
                                    err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
                ax.set_ylabel(type)
                ax.set_xlabel(col)
                ax.set_title(f'Evolution of {type} for task {task}')
                plt.savefig(f"../plot/{folder}/{type}_{col}_{filter_task}_{task}_per_task_lineplot.png")
                plt.show()
            
        for col in ['mean_euclidean_distance', 'mean_delta_successive', 'nb_tentative', 'pct_error']:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            sns.boxplot(x='success', y=col, data=df_grouped_per_student_per_task[df_grouped_per_student_per_task[filter_task] == task], ax=ax)
            ax.set_ylabel(col)
            ax.set_title(f'Boxplot of {col} per student for {task}')
            plt.savefig(f"../plot/{folder}/{col}_{filter_task}_{task}_per_student_boxplot.png")
            plt.show()
            
        # plot delta successive per nb_tentative
        for col in ['mean_euclidean_distance', 'mean_delta_successive', 'pct_error']:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x='nb_tentative', y='mean_delta_successive', hue='success', data=df_grouped_per_student_per_task[df_grouped_per_student_per_task[filter_task] == task], ax=ax, 
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
            ax.set_ylabel(col)
            ax.set_xlabel("Nb of Total Tentative")
            ax.set_title(f'{col} for each Total Tentative for task {task}')
            plt.savefig(f"../plot/{folder}/{col}_{filter_task}_{task}_nb_tentative.png")
            
        
        for col in ['euclidean_distance', 'delta_successive', 'error']:
            
            fig, axs = plt.subplots(4, 3, figsize=(15, 12))
            tot_tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            
            for i, ax in enumerate(axs.flat):
                if i < len(tot_tentatives):
                    grouped_data = df[df[filter_task] == task]
                    grouped_data = grouped_data[grouped_data['nb_tentative'] == tot_tentatives[i]]
                    
                    # count number of failing and successful students
                    nb_fail_student = len(grouped_data[grouped_data['success'] == 0]['Student ID'].unique())
                    nb_success_student = len(grouped_data[grouped_data['success'] == 1]['Student ID'].unique())

                    sns.lineplot(data=grouped_data, y=col, x='index', hue='success',
                                err_style="band", errorbar ='ci', estimator=np.mean, ci=95, ax=ax, legend=True)
                    
                    ax.set_title(f"Total Tentatives: {tot_tentatives[i]}")
                    ax.set_xlabel("Tentative")
                    ax.set_ylabel(col)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    
                    handles, labels = ax.get_legend_handles_labels()
                    new_labels = []
                    for label in labels:
                        if label == '0':
                            new_labels.append(f"Failing, N={nb_fail_student}")
                        else:
                            new_labels.append(f"Successful, N={nb_success_student}")
                    ax.legend(handles, new_labels, loc='upper right', fontsize=9)
                    
                else:
                    ax.axis('off')

            plt.suptitle(f"{col} for task {task}")
            plt.tight_layout()
            plt.savefig(f"../plot/{folder}/{col}_for_task_{task}_per_tentative_multi.png")
            plt.show()
    
                
        # line plot of euclidean distance per task
    for col in ['nb_tentative', 'pct_activity']:
        for type in ['euclidean_distance', 'delta_successive', 'error']:
            fig, ax = plt.subplots(figsize=(10, 5))
            if col == 'nb_tentative':
                sns.lineplot(x=col, y=type, hue=filter_task, data=df, ax=ax,
                                err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
            else: 
                sns.lineplot(x=col, y=type, hue=filter_task, data=df_filtered, ax=ax,
                                err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
            ax.set_ylabel(type)
            ax.set_xlabel(col)
            ax.set_title(f'Evolution of {type} per task')
            plt.savefig(f"../plot/{folder}/{type}_{col}_{filter_task}_per_task_lineplot.png")
            plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for col in ['mean_euclidean_distance', 'mean_delta_successive', 'nb_tentative', 'pct_error']:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            sns.boxplot(x=filter_task, y=col, data=df_grouped_per_student_per_task, ax=ax)
            ax.set_ylabel(col)
            ax.set_title(f'Boxplot of {col} per student per task')
            plt.savefig(f"../plot/{folder}/{col}_{filter_task}_per_student_boxplot.png")
            plt.show()
            
        # plot delta successive per nb_tentative
        for col in ['mean_euclidean_distance', 'mean_delta_successive']:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x='nb_tentative', y='mean_delta_successive', hue=filter_task, data=df_grouped_per_student_per_task, ax=ax, 
                        err_style="band", errorbar ='ci', estimator=np.mean, ci=95, legend=True)
            ax.set_ylabel(f'{col} for each Total Tentative')
            ax.set_xlabel("Nb of Total Tentative")
            ax.set_title(col)
            plt.savefig(f"../plot/{folder}/{col}_{filter_task}_nb_tentative.png")
            
        
def plots_code_space(df, folder, success_filter): 
    
    # plot euclidean distance and Submission_TreeDist_LastSubmission per pct_activity for successful and failing students
    names = ['L1', 'L2', 'L3']
    dfs = [df[(df['activity'] == 1) & (df['success'] == success_filter)], 
           df[(df['activity'] == 2) & (df['success'] == success_filter)], 
           df[(df['activity'] == 3) & (df['success'] == success_filter)]]
    
    label = 'Successful' if success_filter == 1 else 'Failing'
    
    # boxplot of euclidean distance and Submission_TreeDist_LastSubmission
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    sns.boxplot(x='activity', y='delta_successive', data=df, ax=axs[0])
    axs[0].set_ylabel('Delta successive')
    axs[0].set_title(f'Boxplot of successive euclidean distance for {label} students')
    
    sns.boxplot(x='activity', y='Submission_TreeDist_Successive', data=df, ax=axs[1])
    axs[1].set_ylabel('Code space distance')
    axs[1].set_title(f'Boxplot of code space distance for {label} students')
    
    plt.savefig(f"../plot/{folder}/delta_successive_code_space_boxplot_{label}.png")
    plt.show()
    

    for col in ['nb_tentative', 'pct_activity']:
        for df, name in zip(dfs, names):
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            df_filtered = df[df['nb_tentative'] <= 10]
            
            nb_student = len(df_filtered['Student ID'].unique())

            sns.lineplot(data=df_filtered, x=col, y='delta_successive', estimator='mean', ci=95, ax=axs[0])
            sns.lineplot(data=df_filtered, x=col, y='Submission_TreeDist_Successive', estimator='mean', ci=95, ax=axs[1])

            axs[0].set_title(f"Successive Euclidean distance")
            axs[0].set_xlabel(col)

            axs[1].set_title(f"Successive Code space distance")
            axs[1].set_xlabel(col)
            plt.suptitle(f"{name} Task, N={nb_student} {label} students")
            plt.savefig(f"../plot/{folder}/{col}_{name}_euclidean_code_space_{label}.png")
            plt.show()
            
    
    # multi plot of euclidean distance and Submission_TreeDist_LastSubmission per tentative
    tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for data, name in zip(dfs, names):
        for i in range(len(tentatives)):

            data_filtered = data[data['nb_tentative'] <= 10]
            
            data_filtered_tentative = data_filtered[data_filtered['nb_tentative'] == tentatives[i]]
            data_tentative = data_filtered_tentative[data_filtered_tentative['success'] == success_filter]
            
            count_error_per_tentative = data_tentative.groupby('index').agg({'error': 'sum'}).reset_index()
            count_error_per_tentative['success'] = success_filter
            nb_student = len(data_tentative['Student ID'].unique())
            
            if nb_student != 0:
    
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                
                sns.barplot(x='index', y='error', hue='success', data=count_error_per_tentative, ax=axs[0])
                sns.lineplot(data=data_tentative, x='index', y='delta_successive', estimator='mean', ci=95, ax=axs[1], palette='Set2')
                sns.lineplot(data=data_tentative, x='index', y='Submission_TreeDist_Successive', estimator='mean', ci=95, ax=axs[2], palette='Set2')
            
                axs[0].set_xlabel('Tentative')
                axs[0].set_ylabel('Number of error')
                axs[0].set_title('Number of Error per Tentative')

                axs[1].set_title(f"Successive Euclidean distance")
                axs[1].set_xlabel("Tentative")

                axs[2].set_title(f"Successive Code space distance")
                axs[2].set_xlabel("Tentative")
                plt.suptitle(f"{name} Task, Tentative {tentatives[i]}, N={nb_student} {label} students")
                plt.savefig(f"../plot/{folder}/barplot_error_{name}_tentative_{tentatives[i]}_euclidean_code_space_{label}.png")
                plt.show()
         
    # scatter plot of euclidean distance and Submission_TreeDist_Successive
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for ax, df, name in zip(axs, dfs, names):   
        
        df_filtered = df[df['nb_tentative'] <= 10]
        sns.scatterplot(data=df_filtered, x='delta_successive', y='Submission_TreeDist_Successive', hue='success', ax=ax)
        sns.regplot(data=df_filtered, x='delta_successive', y='Submission_TreeDist_Successive', scatter=False, ax=ax)

        ax.set_title(name)

    plt.savefig(f"../plot/{folder}/scatterplot_euclidean_code_space_{label}.png")
    plt.show()

    for df in dfs: 
        print(stats.pearsonr(df['delta_successive'], df['Submission_TreeDist_Successive']))
    

def plots_error_code_world_succ(df, folder, success_filter, type): 
    
    # plot euclidean distance and Submission_TreeDist_LastSubmission per pct_activity for successful and failing students
    names = ['L1', 'L2', 'L3']
    dfs = [df[(df['activity'] == 1) & (df['success'] == success_filter)], 
           df[(df['activity'] == 2) & (df['success'] == success_filter)], 
           df[(df['activity'] == 3) & (df['success'] == success_filter)]]
    
    label = 'Successful' if success_filter == 1 else 'Failing'
    
    # multi plot of euclidean distance and Submission_TreeDist_LastSubmission per tentative
    tentatives = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for data, name in zip(dfs, names):
        for i in range(len(tentatives)):

            data_filtered = data[data['nb_tentative'] <= 10]
            
            data_filtered_tentative = data_filtered[data_filtered['nb_tentative'] == tentatives[i]]
            data_tentative = data_filtered_tentative[data_filtered_tentative['success'] == success_filter]
            
            count_error_per_tentative = data_tentative.groupby(['index', 'Student ID']).agg({'error': 'sum'}).reset_index()
            count_error_per_tentative['success'] = success_filter
            nb_student = len(data_tentative['Student ID'].unique())
            
            if nb_student != 0:
    
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                df_errors = data_tentative[data_tentative['error'] == 1]
                # Create a count of errors by index and Student ID for stacking
                df_pivot = df_errors.pivot_table(index='index', columns='Student ID', values='error', aggfunc='sum', fill_value=0)
                #print(df_pivot)
                
                if not df_pivot.empty:
                    unique_students = df_errors['Student ID'].unique()
                    palette = sns.color_palette("tab20", len(unique_students))
                    student_colors = {student: palette[i] for i, student in enumerate(unique_students)}

                    # Plot the stacked bar chart
                    df_pivot.plot(kind='bar', stacked=True, ax=axs[0], legend=False, color=[student_colors[col] for col in df_pivot.columns])
                #sns.barplot(x='index', y='error', hue='Student ID', data=data_tentative, ax=axs[0], legend=False, palette='muted')
                sns.lineplot(data=data_tentative, x='index', y=f'{type}_delta_successive', estimator='mean', ci=95, ax=axs[1], palette='Set2')
                sns.lineplot(data=data_tentative, x='index', y=f'{type}_Submission_TreeDist_Successive', estimator='mean', ci=95, ax=axs[2], palette='Set2')
            
                axs[0].set_xlabel('Tentative')
                axs[0].set_ylabel('Number of error')
                axs[0].set_title('Number of Error per Tentative')

                axs[1].set_title(f'{type}_delta_successive')
                axs[1].set_xlabel("Tentative")

                axs[2].set_title(f'{type}_Submission_TreeDist_Successive')
                axs[2].set_xlabel("Tentative")
                plt.suptitle(f"{name} Task, Tentative {tentatives[i]}, N={nb_student} {label} students")
                plt.savefig(f"../plot/{folder}/barplot_error_per_student_{name}_tentative_{tentatives[i]}_euclidean_code_space_{label}.png")
                plt.show()
        
        
def plot_loss_roc(loss_train, loss_valid, roc_auc, fpr, tpr, roc=False): 
    # Plot ROC curve
    if roc:
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
    # Plot loss
    plt.plot(loss_train, label='Train')
    plt.plot(loss_valid, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
