import matplotlib.pyplot as plt
from tests import *
from distances import *


### For the tests 

def test_plot_one_metric(metric_type, list_vec, goal_vec, title, test=False):
    # Calculate the length of the list_vec
    length = len(list_vec)
    
    # Create indices for x-axis
    indices = list(range(1, length + 1))
    
    # Dictionary to map metric types to corresponding test functions
    test_functions = {
        "levenshtein": test_levenshtein,
        "ratcliff": test_ratcliff,
        "jaro": test_jaro,
        "euclidean": test_euclidean
    }
    
    # Check if metric_type is valid
    if metric_type not in test_functions:
        raise ValueError("Invalid metric type:", metric_type)
    
    # Get the appropriate test function
    test_func = test_functions[metric_type]
    
    # Get results for different configurations
    
    if metric_type != "euclidean":
        results = {}
        configurations = ["row", "col", "bars", "row_concat", "col_concat"]
        for config in configurations:
            results[config] = test_func(config, list_vec, goal_vec, test=test)
    else:
        results = test_func(list_vec, goal_vec, test=test)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    for config, result in results.items():
        plt.plot(indices, result, label=config, alpha=0.5, marker='o')
    
    # Set plot title and labels
    plt.title(title)
    plt.ylabel(metric_type + " distance")
    plt.xlabel("Indices")
    plt.legend()
    

def test_comparing_metrics(list_vec, goal_vec, title, print_results=False, test=False):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    levenshtein_results_row = test_levenshtein("row", list_vec, goal_vec, test=test)
    levenshtein_results_column = test_levenshtein("col", list_vec, goal_vec, test=test)
    levenshtein_results_bars = test_levenshtein("bars", list_vec, goal_vec, test=test)
    levenshtein_results_row_concat = test_levenshtein("row_concat", list_vec, goal_vec, test=test)
    levenshtein_results_col_concat = test_levenshtein("col_concat", list_vec, goal_vec, test=test)

    ax[0, 0].plot(indices, levenshtein_results_row, label="per row", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_column, label="per column", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_bars, label="per bar", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_col_concat, label="col concat", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_row_concat, label="row concat", alpha=0.5, marker='o')
    ax[0, 0].set_title("Levenshtein distance")
    ax[0, 0].set_ylabel("Levenshtein distance")
    ax[0, 0].legend()

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results, test=test)

    ax[0, 1].plot(indices, euclidean_results, label="Euclidean", alpha=0.5, marker='o')
    ax[0, 1].set_title("Euclidean distance")
    ax[0, 1].set_ylabel("Euclidean distance")
    ax[0, 1].legend()

    ratcliff_results_row = test_ratcliff("row", list_vec, goal_vec, test=test)
    ratcliff_results_column = test_ratcliff("col", list_vec, goal_vec, test=test)
    ratcliff_results_bars = test_ratcliff("bars", list_vec, goal_vec, test=test)
    ratcliff_results_row_concat = test_ratcliff("row_concat", list_vec, goal_vec, test=test)
    ratcliff_results_col_concat = test_ratcliff("col_concat", list_vec, goal_vec, test=test)

    ax[1, 0].plot(indices, ratcliff_results_row, label="per row", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_column, label="per column", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_bars, label="per bar", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_col_concat, label="col concat", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_row_concat, label="row concat", alpha=0.5, marker='o')
    ax[1, 0].set_title("Ratcliff distance")
    ax[1, 0].set_ylabel("Ratcliff distance")
    ax[1, 0].legend()

    jaro_results_row = test_jaro("row", list_vec, goal_vec, test=test)
    jaro_results_column = test_jaro("col", list_vec, goal_vec, test=test)
    jaro_results_bars = test_jaro("bars", list_vec, goal_vec, test=test)
    jaro_results_row_concat = test_jaro("row_concat", list_vec, goal_vec, test=test)
    jaro_results_col_concat = test_jaro("col_concat", list_vec, goal_vec, test=test)

    ax[1, 1].plot(indices, jaro_results_row, label="per row", alpha=0.5, marker='o')
    ax[1, 1].plot(indices, jaro_results_column, label="per column", alpha=0.5, marker='o')
    ax[1, 1].plot(indices, jaro_results_bars, label="per bar", alpha=0.5, marker='o')
    ax[1, 1].plot(indices, jaro_results_col_concat, label="col concat", alpha=0.5, marker='o')
    ax[1, 1].plot(indices, jaro_results_row_concat, label="row concat", alpha=0.5, marker='o')
    ax[1, 1].set_title("Jaro distance")
    ax[1, 1].set_ylabel("Jaro distance")
    ax[1, 1].legend()

    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    plt.show()
    
def test_comparing_euclidean_jaro(list_vec, goal_vec, title, print_results=False, test=False):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    jaro_results_bars = test_jaro("bars", list_vec, goal_vec , test=test)
    jaro_results_col_concat = test_jaro("col_concat", list_vec, goal_vec, test=test)

    ax[0].plot(indices, jaro_results_bars, label="bars", alpha=0.5, marker='o')
    ax[0].plot(indices, jaro_results_col_concat, label="column concatenated", alpha=0.5, marker='o')
    ax[0].set_title("Jaro distance")
    ax[0].set_ylabel("Jaro score")
    ax[0].legend()

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results, test=test)

    ax[1].plot(indices, euclidean_results, label="Euclidean", alpha=0.5, marker='o')
    ax[1].set_title("Euclidean distance")
    ax[1].set_ylabel("Euclidean distance")
    ax[1].legend()
    
    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    plt.show()
    
    
def plot_comparing_euclidean(dict_vec_success, dict_vec_fail, goal_vec, title, print_results=False, test=False):
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    for student in dict_vec_success:
        list_vec = dict_vec_success[student]
        length = len(list_vec)
        indices = [i + 1 for i in range(length)]

        euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results, test=test)

        ax[0].plot(indices, euclidean_results, label=student, alpha=0.5, marker='o')
        ax[0].set_title("Euclidean distance for successful students")
        ax[0].set_ylabel("Euclidean distance")
        ax[0].legend(loc='upper right')
        
    for student in dict_vec_fail:
        list_vec = dict_vec_fail[student]
        length = len(list_vec)
        indices = [i + 1 for i in range(length)]

        euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results, test=test)

        ax[1].plot(indices, euclidean_results, label=student, alpha=0.5, marker='o')
        ax[1].set_title("Euclidean distance for unsuccessful students")
        ax[1].set_ylabel("Euclidean distance")
        ax[1].legend(loc='upper right')

    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    plt.show()
    

def plot_single_euclidean(list_vec, goal_vec, title, print_results=False, test=False):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results, test=test)

    ax.plot(indices, euclidean_results, alpha=0.5, marker='o')
    ax.set_ylabel("Euclidean distance")
    
    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    plt.show()