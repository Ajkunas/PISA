from distances import *
import matplotlib.pyplot as plt

##### Basic tests for the Learning task 1 #####

KEY_VECTOR_V = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

# Initial position
V1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"

# Cans are next to each other but in the wrong position (next to initial position)
V2 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,E,E,E,false']"

# Cans are shifted to the left, from one position from the initial position
V3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,false']"

# Cans next to each other, shifted from one position from the initial position
V4 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,E,E,false']"

# Cans are shifted to the left, from one positions from the initial position
V5 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,false']"

# One can in good position, the other is not: shifted to the left by one position from the good one (next to each other)
V6 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,false']"

# Both cans are in good position
V7 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is at the initial position
V8 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is shifted to the right from the initial position
V9 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is shifted to the right from the initial position
V10 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,ra-world-shape ra-world-shapeA,false']"

# Cans are shifted to the left, from two positions from the initial position
V11 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,false']"

# Comparing vectors from initial state to the goal state by moving one can at a time (from left to right)
LIST_VECTORS_1 = [V1, V2, V3, V4, V5, V6, V7]

# Varying the position of one of the can, while the other is in good position
LIST_VECTORS_2 = [V8, V9, V10, V6]

# Varying the shift of both cans 
LIST_VECTORS_3 = [V1, V3, V11, V5, V7]



############ BASIC TESTS FOR CASE WITH 1 CAN AND 1 GLASS #############

KEY_VECTOR_W = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,false']"

### TEST : going toward the goal state step by step ###

# Initial state where can and glass inversed positions and at the other end
W1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeB,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"

W2 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,E,E,false']"

W3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,E,E,false']"

W4 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,E,E,false']"

W5 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,false']"

W6 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,false']"

W7 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,E,E,ra-world-shape ra-world-shapeB,E,false']"

W8 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"

W9 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

W10 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,false']"

W11 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,false']"

LIST_VECTOR_W = [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11]
    
    
    
### TEST : testing the variation of shifts and order towars the goal state ###
# initial state
W12 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeB,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"

W13 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,', 'ra-world-shape ra-world-shapeB,E,E,E,E,false']"
W14 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,E,E,false']"
W15 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,false']"
W16 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,false']"
W17 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,false']"
W18 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,E,E,ra-world-shape ra-world-shapeB,E,false']"
W19 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeB,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,false']"
W20 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"
W21 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,false']"

LIST_VECTOR_W2 = [W12, W13, W14, W15, W16, W17, W18, W19, W20, W21]

############ BASIC TESTS FOR CASE WITH 2 CANS AND 2 GLASSES - L2 task #############
KEY_VECTOR_X = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,false']"

# Initial state
X1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,false']"

## Cases where the student does everything right
X2 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,E,E,false']"
X3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,E,false']"
X4 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,E,false']"
X5 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,false']"

LIST_VECTOR_X1 = [X1, X2, X3, X4, X5]

## Testing cases where the student makes mistakes
X6 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeB,false']"
X7 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,E,false']"
X8 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeB,E,E,false']"
X9 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,E,E,false']"

LIST_VECTOR_X2 = [X6, X7, X8, X9]


############ BASIC TESTS FOR CASE WITH 2 CANS AND 2 GLASSES #############
KEY_VECTOR_Z = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"

# Initial state 
Z1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,false']"
Z2 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"
Z3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"
Z4 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeB,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,false']"
Z5 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,false']"
Z6 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"

LIST_VECTOR_Z1 = [Z1, Z2, Z3, Z4, Z5, Z6]
    
########### COMPARING THE DISTANCES ###########

### For Levenshtein distance ###
def test_levenshtein(metric_type, list_vec, goal_vec, print_results=False):
    if print_results:
        print("Levenshtein distance for", metric_type)
    results = []
    
    for v in list_vec:
        distance = levenshtein_distance_combined(v, goal_vec, metric_type)
        results.append(distance)
        
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Ratcliff distance ###
def test_ratcliff(metric_type, list_vec, goal_vec, print_results=False):
    
    if print_results:
        print("Ratcliff distance for", metric_type)
    results = []
    
    for v in list_vec:
        distance = ratcliff_obershelp_distance_combined(v, goal_vec, metric_type)
        results.append(distance)
        
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Jaro distance ###

def test_jaro(metric_type, list_vec, goal_vec, print_results=False):
    
    if print_results:
        print("Jaro distance for", metric_type)
    results = []
    
    for v in list_vec:
        distance = jaro_combined(v, goal_vec, method=metric_type)
        results.append(distance)
        
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Jaro-Winkler distance ###
def test_jaro_winkler(metric_type, list_vec, goal_vec, print_results=False):
    
    if print_results:
        print("Jaro-Winkler distance for", metric_type)
    results = []
    
    for v in list_vec:
        distance = jaro_combined(v, goal_vec, similarity_metric='jaro_winkler', method=metric_type)
        results.append(distance)
        
        if print_results:
            print("Vector:", v)
            print(distance)
    return results
            
### For Euclidean distance ###
def test_euclidean(list_vec, goal_vec, print_results=False):
    no_penalties = {'move1': 0, 'move2': 0, 'pickup1': 0, 'place1': 0}
    
    if print_results:
        print("Euclidean distance")
    results = []
    
    for v in list_vec:
        distance = euclidean_v2(v, goal_vec, move1=0, move2=0, place1=0, pickup1=0, missing=0, penalties=no_penalties)
        results.append(distance)
        
        if print_results:
            print("Vector:", v)
            print(distance)
            
    return results



#################### plot the results ####################

def test_plot_one_metric(metric_type, list_vec, goal_vec, title, nb, folder_name):
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
            results[config] = test_func(config, list_vec, goal_vec)
    else:
        results = test_func(list_vec, goal_vec)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    for config, result in results.items():
        plt.plot(indices, result, label=config, alpha=0.5, marker='o')
    
    # Set plot title and labels
    plt.title(title)
    plt.ylabel(metric_type + " distance")
    plt.xlabel("Indices")
    plt.legend()
    plt.savefig(folder_name + metric_type + "_list_" + str(nb) + ".png")
    plt.show()
    

def test_comparing_metrics(list_vec, goal_vec, title, folder_name, nb, print_results=False):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    levenshtein_results_row = test_levenshtein("row", list_vec, goal_vec)
    levenshtein_results_column = test_levenshtein("col", list_vec, goal_vec)
    levenshtein_results_bars = test_levenshtein("bars", list_vec, goal_vec)
    levenshtein_results_row_concat = test_levenshtein("row_concat", list_vec, goal_vec)
    levenshtein_results_col_concat = test_levenshtein("col_concat", list_vec, goal_vec)

    ax[0, 0].plot(indices, levenshtein_results_row, label="per row", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_column, label="per column", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_bars, label="per bar", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_col_concat, label="col concat", alpha=0.5, marker='o')
    ax[0, 0].plot(indices, levenshtein_results_row_concat, label="row concat", alpha=0.5, marker='o')
    ax[0, 0].set_title("Levenshtein distance")
    ax[0, 0].set_ylabel("Levenshtein distance")
    ax[0, 0].legend()

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results)

    ax[0, 1].plot(indices, euclidean_results, label="Euclidean", alpha=0.5, marker='o')
    ax[0, 1].set_title("Euclidean distance")
    ax[0, 1].set_ylabel("Euclidean distance")
    ax[0, 1].legend()

    ratcliff_results_row = test_ratcliff("row", list_vec, goal_vec)
    ratcliff_results_column = test_ratcliff("col", list_vec, goal_vec)
    ratcliff_results_bars = test_ratcliff("bars", list_vec, goal_vec)
    ratcliff_results_row_concat = test_ratcliff("row_concat", list_vec, goal_vec)
    ratcliff_results_col_concat = test_ratcliff("col_concat", list_vec, goal_vec)

    ax[1, 0].plot(indices, ratcliff_results_row, label="per row", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_column, label="per column", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_bars, label="per bar", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_col_concat, label="col concat", alpha=0.5, marker='o')
    ax[1, 0].plot(indices, ratcliff_results_row_concat, label="row concat", alpha=0.5, marker='o')
    ax[1, 0].set_title("Ratcliff distance")
    ax[1, 0].set_ylabel("Ratcliff distance")
    ax[1, 0].legend()
    
    jaro_results_row = test_jaro("row", list_vec, goal_vec)
    jaro_results_column = test_jaro("col", list_vec, goal_vec)
    jaro_results_bars = test_jaro("bars", list_vec, goal_vec)
    jaro_results_row_concat = test_jaro("row_concat", list_vec, goal_vec)
    jaro_results_col_concat = test_jaro("col_concat", list_vec, goal_vec)

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
    
    plt.savefig(folder_name + "comparison_metric_list_" + str(nb) + ".png")
    plt.show()
    
def test_comparing_euclidean_jaro(list_vec, goal_vec, title, folder_name, nb, print_results=False):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    jaro_results_bars = test_jaro("bars", list_vec, goal_vec)
    jaro_results_col_concat = test_jaro("col_concat", list_vec, goal_vec)

    ax[0].plot(indices, jaro_results_bars, label="bars", alpha=0.5, marker='o')
    ax[0].plot(indices, jaro_results_col_concat, label="column concatenated", alpha=0.5, marker='o')
    ax[0].set_title("Jaro distance")
    ax[0].set_ylabel("Jaro score")
    ax[0].legend()

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results)

    ax[1].plot(indices, euclidean_results, label="Euclidean", alpha=0.5, marker='o')
    ax[1].set_title("Euclidean distance")
    ax[1].set_ylabel("Euclidean distance")
    ax[1].legend()
    
    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    plt.savefig(folder_name + "comparison_euclidean_jaro_list_" + str(nb) + ".png")
    plt.show()
    
    
def plot_comparing_euclidean(dict_vec_success, dict_vec_fail, goal_vec, title, folder_name, nb, print_results=False):
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    for student in dict_vec_success:
        list_vec = dict_vec_success[student]
        length = len(list_vec)
        indices = [i + 1 for i in range(length)]

        euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results)

        ax[0].plot(indices, euclidean_results, label=student, alpha=0.5, marker='o')
        ax[0].set_title("Euclidean distance for successful students")
        ax[0].set_ylabel("Euclidean distance")
        ax[0].legend(loc='upper right')
        
    for student in dict_vec_fail:
        list_vec = dict_vec_fail[student]
        length = len(list_vec)
        indices = [i + 1 for i in range(length)]

        euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results)

        ax[1].plot(indices, euclidean_results, label=student, alpha=0.5, marker='o')
        ax[1].set_title("Euclidean distance for unsuccessful students")
        ax[1].set_ylabel("Euclidean distance")
        ax[1].legend(loc='upper right')

    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    
    plt.savefig(folder_name + "comparison_euclidean_list_" + str(nb) + ".png")
    plt.show()
    

def plot_single_euclidean(list_vec, goal_vec, title, folder_name, nb, print_results=False):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    length = len(list_vec)
    indices = [i + 1 for i in range(length)]

    euclidean_results = test_euclidean(list_vec, goal_vec, print_results=print_results)

    ax.plot(indices, euclidean_results, alpha=0.5, marker='o')
    ax.set_ylabel("Euclidean distance")
    
    # title for the whole plot
    fig.suptitle(title)
    plt.legend()
    
    plt.savefig(folder_name + "euclidean_list_" + str(nb) + ".png")
    plt.show()

