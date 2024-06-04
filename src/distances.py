# File with the distances metrics for the world space, for the RobotArm activity.
import numpy as np
import ast
import textdistance as td
from itertools import permutations

def preprocess(vector): 
    vector = vector.replace("ra-world-shape ra-world-shapeA", "A")
    vector = vector.replace("ra-world-shape ra-world-shapeB", "B")
    vector = ast.literal_eval(vector)
    
    matrix = []
    for i in range(len(vector)):
        matrix.append(vector[i].split(','))
    
    # drops first row
    matrix = matrix[1:]
        
    for i in range(len(matrix)):
        matrix[i] = matrix[i][:-1]
        
    return np.array(matrix)

# https://machinelearningknowledge.ai/ways-to-calculate-levenshtein-distance-edit-distance-in-python/
def levenshtein_distance(s, t):
    m = len(s)
    n = len(t)
    d = [[0] * (n + 1) for i in range(m + 1)]  

    for i in range(1, m + 1):
        d[i][0] = i

    for j in range(1, n + 1):
        d[0][j] = j
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                          d[i][j - 1] + 1,      # insertion
                          d[i - 1][j - 1] + cost) # substitution   

    return d[m][n]

def levenshtein_distance_combined(v1, v2, method='row_concat', test=False):
    if not test:
        v1_matrix = preprocess(v1)
        v2_matrix = preprocess(v2)
    else: 
        v1_matrix = v1
        v2_matrix = v2
    
    if method == 'row_concat':
        v1_string = ''.join([''.join(row) for row in v1_matrix])
        v2_string = ''.join([''.join(row) for row in v2_matrix])
    elif method == 'col_concat':
        v1_string = ''.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = ''.join([''.join(col) for col in zip(*v2_matrix)])
    elif method == 'row':
        distance = 0
        for row1, row2 in zip(v1_matrix, v2_matrix):
            v1_string = ''.join(row1)
            v2_string = ''.join(row2)
            distance += levenshtein_distance(v1_string, v2_string) / len(v1_string)
        return distance / v1_matrix.shape[0]
    elif method == 'col':
        distance = 0
        for col1, col2 in zip(zip(*v1_matrix), zip(*v2_matrix)):
            v1_string = ''.join(col1)
            v2_string = ''.join(col2)
            distance += levenshtein_distance(v1_string, v2_string) / len(v1_string)
        return distance / v1_matrix.shape[1]
    elif method == 'bars':
        v1_string = '|'.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = '|'.join([''.join(col) for col in zip(*v2_matrix)])
    else:
        raise ValueError("Invalid method provided")

    return levenshtein_distance(v1_string, v2_string) / len(v1_string)

###### Ratcliff Obershelp Similarity ######
def ratcliff_obershelp_distance_combined(v1, v2, method='row_concat', test=False):
    if not test:
        v1_matrix = preprocess(v1)
        v2_matrix = preprocess(v2)
    else: 
        v1_matrix = v1
        v2_matrix = v2
    
    if method == 'row_concat':
        v1_string = ''.join([''.join(row) for row in v1_matrix])
        v2_string = ''.join([''.join(row) for row in v2_matrix])
    elif method == 'col_concat':
        v1_string = ''.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = ''.join([''.join(col) for col in zip(*v2_matrix)])
    elif method == 'row':
        distance = 0
        for row1, row2 in zip(v1_matrix, v2_matrix):
            v1_string = ''.join(row1)
            v2_string = ''.join(row2)
            distance += 1 - td.ratcliff_obershelp.normalized_similarity(v1_string, v2_string)
        return distance / v1_matrix.shape[0]
    elif method == 'col':
        distance = 0
        for col1, col2 in zip(zip(*v1_matrix), zip(*v2_matrix)):
            v1_string = ''.join(col1)
            v2_string = ''.join(col2)
            distance += 1 - td.ratcliff_obershelp.normalized_similarity(v1_string, v2_string)
        return distance / v1_matrix.shape[1]
    elif method == 'bars':
        v1_string = '|'.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = '|'.join([''.join(col) for col in zip(*v2_matrix)])
    else:
        raise ValueError("Invalid method provided")

    return 1 - td.ratcliff_obershelp.normalized_similarity(v1_string, v2_string)

# Simple metric, comparing the strings of the vectors
def simple_metric(v1, v2):
    v1_matrix = preprocess(v1).T
    v2_matrix = preprocess(v2).T
    
    v1_string = ''
    for i in range(len(v1_matrix)):
        v1_string += ''.join(v1_matrix[i])
        
    v2_string = ''
    for i in range(len(v2_matrix)):
        v2_string += ''.join(v2_matrix[i])
    
    if len(v1_string) != len(v2_string):
        return -1

    distance = 0
    
    for i in range(len(v1_string)):
        if v1_string[i] != v2_string[i]:
            distance += 1
        
    return (distance / len(v1_string))

################### Jaro and Jaro-Winkler Similarity ##################

#https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/

# Function to calculate the 
# Jaro Similarity of two strings 
def jaro_distance(s1, s2):

	# If the strings are equal 
	if (s1 == s2):
		return 1.0

	# Length of two strings 
	len1 = len(s1)
	len2 = len(s2)

	if (len1 == 0 or len2 == 0):
		return 0.0

	# Maximum distance upto which matching 
	# is allowed 
	max_dist = (max(len(s1), len(s2)) // 2 ) - 1

	# Count of matches 
	match = 0

	# Hash for matches 
	hash_s1 = [0] * len(s1)
	hash_s2 = [0] * len(s2)

	# Traverse through the first string 
	for i in range(len1): 

		# Check if there is any matches 
		for j in range( max(0, i - max_dist), 
					min(len2, i + max_dist + 1)): 
			
			# If there is a match 
			if (s1[i] == s2[j] and hash_s2[j] == 0): 
				hash_s1[i] = 1
				hash_s2[j] = 1
				match += 1
				break; 
		
	# If there is no match 
	if (match == 0):
		return 0.0

	# Number of transpositions 
	t = 0

	point = 0

	# Count number of occurrences 
	# where two characters match but 
	# there is a third matched character 
	# in between the indices 
	for i in range(len1): 
		if (hash_s1[i]):

			# Find the next matched character 
			# in second string 
			while (hash_s2[point] == 0):
				point += 1

			if (s1[i] != s2[point]):
				point += 1
				t += 1
			else:
				point += 1
				
		t /= 2

	# Return the Jaro Similarity 
	return ((match / len1 + match / len2 +
			(match - t) / match ) / 3.0)

# Jaro Winkler Similarity 
def jaro_Winkler(s1, s2): 

	jaro_dist = jaro_distance(s1, s2); 

	# If the jaro Similarity is above a threshold 
	if (jaro_dist > 0.7):

		# Find the length of common prefix 
		prefix = 0

		for i in range(min(len(s1), len(s2))):
		
			# If the characters match 
			if (s1[i] == s2[i]):
				prefix += 1

			# Else break 
			else :
				break

		# Maximum of 4 characters are allowed in prefix 
		prefix = min(4, prefix)

		# Calculate jaro winkler Similarity 
		jaro_dist += 0.1 * prefix * (1 - jaro_dist)

	return jaro_dist

def similarity_score(string1, string2, method):
    if method == 'jaro':
        return td.jaro.normalized_similarity(string1, string2)
    elif method == 'jaro_winkler':
        return td.jaro_winkler.normalized_similarity(string1, string2)
    else:
        raise ValueError("Invalid method provided")

def jaro_combined(v1, v2, similarity_metric='jaro', method='row_concat', test=False):
    if not test:
        v1_matrix = preprocess(v1)
        v2_matrix = preprocess(v2)
    else: 
        v1_matrix = v1
        v2_matrix = v2
    
    if method == 'row_concat':
        v1_string = ''.join([''.join(row) for row in v1_matrix])
        v2_string = ''.join([''.join(row) for row in v2_matrix])
    elif method == 'col_concat':
        v1_string = ''.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = ''.join([''.join(col) for col in zip(*v2_matrix)])
    elif method == 'bars':
        v1_string = '|'.join([''.join(col) for col in zip(*v1_matrix)])
        v2_string = '|'.join([''.join(col) for col in zip(*v2_matrix)])
    elif method == 'col':
        distance = 0
        for col1, col2 in zip(zip(*v1_matrix), zip(*v2_matrix)):
            v1_string = ''.join(col1)
            v2_string = ''.join(col2)
            distance += 1 - similarity_score(v1_string, v2_string, similarity_metric)
        return distance / v1_matrix.shape[1]
    elif method == 'row':
        distance = 0
        for row1, row2 in zip(v1_matrix, v2_matrix):
            v1_string = ''.join(row1)
            v2_string = ''.join(row2)
            distance += 1 - similarity_score(v1_string, v2_string, similarity_metric)
        return distance / v1_matrix.shape[0]
    else:
        raise ValueError("Invalid method provided")

    return 1 - similarity_score(v1_string, v2_string, similarity_metric)

### Euclidean distance ###

# when student does an error => how to penalize it ? recomputing initial distance or giving the previous distance ? 
# if we give the previous distance, we need to store it in the state

# important to have the same number of elements in the two matrices => if it is not the case => what to return ? initial distance ? 

def euclidean_v1(v, goal_v, print_results=False, penalty=True):
    
    # to tune !! 
    ERROR_PENALTY = 0.1
    ERROR = False
    tot_distance = 0
    
    v_matrix = None
    goal_matrix = None
    
        
    v = v.replace("ra-world-shape ra-world-shapeA", "A")
    v = v.replace("ra-world-shape ra-world-shapeB", "B")
    v = ast.literal_eval(v)
    
    matrix = []
    for i in range(len(v)):
        matrix.append(v[i].split(','))
    
    # drops first row
    matrix = matrix[1:]
        
    if matrix[-1][-1] != 'false':
        ERROR = True
        
    for i in range(len(matrix)):
        matrix[i] = matrix[i][:-1]
    
    v_matrix = np.array(matrix)
    
    goal_matrix = preprocess(goal_v)
        
    #print(v_matrix)
    #print(goal_matrix)
    
    # count number of elements A in the matrix
    initial_count_A = np.count_nonzero(v_matrix == 'A')
    initial_count_B = np.count_nonzero(v_matrix == 'B')
    #print("Initial count A:", initial_count_A)
    #print("Initial count B:", initial_count_B)
    
    goal_count_A = np.count_nonzero(goal_matrix == 'A')
    goal_count_B = np.count_nonzero(goal_matrix == 'B')
    #print("Goal count A:", goal_count_A)
    #print("Goal count B:", goal_count_B)
        
    initial_indices_A = np.argwhere(v_matrix == 'A')
    initial_indices_B = np.argwhere(v_matrix == 'B')
    goal_indices_A = np.argwhere(goal_matrix == 'A')
    goal_indices_B = np.argwhere(goal_matrix == 'B')

    init_permutations_indices_A = list(permutations(initial_indices_A))
    init_permutations_indices_B = list(permutations(initial_indices_B))
    goal_permutations_indices_A = list(permutations(goal_indices_A))
    goal_permutations_indices_B = list(permutations(goal_indices_B))
    
    distances_A = []
    distances_B = []
    
    # Calculate the maximum possible Euclidean distance
    max_distance = np.linalg.norm(np.array([0, 0]) - np.array([len(goal_matrix)-1, len(goal_matrix[0])-1]))
    #print("Max distance:", max_distance)

    # for A
    for goal_perm_A in goal_permutations_indices_A:
        for init_perm_A in init_permutations_indices_A:
            # Calculate Euclidean distance for the current permutation
            distance_A = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_A), np.array(init_perm_A))]
            distance_A = sum(distance_A)
            distances_A.append(distance_A)
            
    # for B
    for goal_perm_B in goal_permutations_indices_B:
        for init_perm_B in init_permutations_indices_B:
            # Calculate Euclidean distance for the current permutation
            distance_B = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_B), np.array(init_perm_B))]
            distance_B = sum(distance_B)
            distances_B.append(distance_B)
            
            
    #print("Distances A:", distances_A)
    #print("Distances B:", distances_B)
    
    min_distance_A = min(distances_A)
    min_distance_B = min(distances_B)
    #print("Min distance A:", min_distance_A)
    #print("Min distance B:", min_distance_B)
    
    # Normalize distances by dividing by the maximum distance
    normalized_distance_A = min_distance_A / max_distance if max_distance != 0 else 0
    normalized_distance_B = min_distance_B / max_distance if max_distance != 0 else 0
    
    #print("Normalized distance A:", normalized_distance_A)
    #print("Normalized distance B:", normalized_distance_B)
    
    # count how many elements are not equal to E in the v_matrix
    count = np.count_nonzero(v_matrix != 'E')
    #print("Count:", count)

    tot_distance = (normalized_distance_A + normalized_distance_B) / count
    
    if print_results:
        print("Minimum distance for A:", min_distance_A)
        print("Minimum distance for B:", min_distance_B)
        print("Total distance:", tot_distance)
        
    # How to penalize errors ?
    if penalty: 
        if not ERROR:
            if initial_count_A != goal_count_A or initial_count_B != goal_count_B:
                tot_distance += ERROR_PENALTY

        else: 
            tot_distance += ERROR_PENALTY
    
    return tot_distance



################################################################################################
# Euclidean distance after some error evaluation 

def euclidean_distance(v_matrix, goal_matrix): 
    
    tot_distance = 0
        
    initial_indices_A = np.argwhere(v_matrix == 'A')
    initial_indices_B = np.argwhere(v_matrix == 'B')
    goal_indices_A = np.argwhere(goal_matrix == 'A')
    goal_indices_B = np.argwhere(goal_matrix == 'B')

    init_permutations_indices_A = permutations(initial_indices_A)
    init_permutations_indices_B = permutations(initial_indices_B)
    goal_permutations_indices_A = permutations(goal_indices_A)
    goal_permutations_indices_B = permutations(goal_indices_B)

    init_permutations_indices_A = list(init_permutations_indices_A)
    init_permutations_indices_B = list(init_permutations_indices_B)
    goal_permutations_indices_A = list(goal_permutations_indices_A)
    goal_permutations_indices_B = list(goal_permutations_indices_B)
    
    distances_A = []
    distances_B = []
    
    # Calculate the maximum possible Euclidean distance
    max_distance = np.linalg.norm(np.array([0, 0]) - np.array([len(goal_matrix)-1, len(goal_matrix[0])-1]))

    # for A
    for goal_perm_A in goal_permutations_indices_A:
        for init_perm_A in init_permutations_indices_A:
            # Calculate Euclidean distance for the current permutation
            distance_A = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_A), np.array(init_perm_A))]
            distance_A = sum(distance_A)
            distances_A.append(distance_A)
            
    # for B
    for goal_perm_B in goal_permutations_indices_B:
        for init_perm_B in init_permutations_indices_B:
            # Calculate Euclidean distance for the current permutation
            distance_B = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_B), np.array(init_perm_B))]
            distance_B = sum(distance_B)
            distances_B.append(distance_B)
            
            
    min_distance_A = min(distances_A)
    min_distance_B = min(distances_B)
    
    # Normalize distances by dividing by the maximum distance
    normalized_distance_A = min_distance_A / max_distance if max_distance != 0 else 0
    normalized_distance_B = min_distance_B / max_distance if max_distance != 0 else 0
    
    # count how many elements are not equal to E in the v_matrix
    count = np.count_nonzero(v_matrix == 'A') + np.count_nonzero(v_matrix == 'B')
    print("Count :", count)

    tot_distance = (normalized_distance_A + normalized_distance_B) / count
    
    return tot_distance


def euclidean(v, goal_v, move1, move2, place1, pickup1, missing, penalties):
    
    tot_distance = 0
        
    v_matrix = preprocess(v)
    goal_matrix = preprocess(goal_v)
    
    if missing == 1: 
        missing_element = None
        
        initial_count_A = np.count_nonzero(v_matrix == 'A')
        initial_count_B = np.count_nonzero(v_matrix == 'B')
        
        goal_count_A = np.count_nonzero(goal_matrix == 'A')
        goal_count_B = np.count_nonzero(goal_matrix == 'B')
        
        if initial_count_A < goal_count_A:
            missing_element = 'A'
        elif initial_count_B < goal_count_B:
            missing_element = 'B'
            
        if missing_element is not None:
            v = v.replace("ra-world-arm", missing_element)
        
        v_matrix = preprocess(v)
        
        tot_distance += euclidean_distance(v_matrix, goal_matrix)
        
        if move1 == 1:
            tot_distance += penalties['move1']
        elif pickup1 == 1:
            tot_distance += penalties['pickup1']
        elif move2 == 1:
            tot_distance += penalties['move2']
    
    else: 
        tot_distance += euclidean_distance(v_matrix, goal_matrix)
        
        if move1 == 1:
            tot_distance += penalties['move1']
        elif place1 == 1:
            tot_distance += penalties['place1']
        elif move2 == 1: 
            tot_distance += penalties['move2']
    
    return tot_distance

######################################## TESTING ANOTHER VERSION WHERE NORMALIZATION IS DONE AT THE END

def euclidean_distance_v2(v_matrix, goal_matrix): 
    
    tot_distance = 0
        
    initial_indices_A = np.argwhere(v_matrix == 'A')
    initial_indices_B = np.argwhere(v_matrix == 'B')
    goal_indices_A = np.argwhere(goal_matrix == 'A')
    goal_indices_B = np.argwhere(goal_matrix == 'B')

    init_permutations_indices_A = permutations(initial_indices_A)
    init_permutations_indices_B = permutations(initial_indices_B)
    goal_permutations_indices_A = permutations(goal_indices_A)
    goal_permutations_indices_B = permutations(goal_indices_B)

    init_permutations_indices_A = list(init_permutations_indices_A)
    init_permutations_indices_B = list(init_permutations_indices_B)
    goal_permutations_indices_A = list(goal_permutations_indices_A)
    goal_permutations_indices_B = list(goal_permutations_indices_B)
    
    distances_A = []
    distances_B = []
    
    # Calculate the maximum possible Euclidean distance
    #max_distance = np.linalg.norm(np.array([0, 0]) - np.array([len(goal_matrix)-1, len(goal_matrix[0])-1]))

    # for A
    for goal_perm_A in goal_permutations_indices_A:
        for init_perm_A in init_permutations_indices_A:
            # Calculate Euclidean distance for the current permutation
            distance_A = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_A), np.array(init_perm_A))]
            distance_A = sum(distance_A)
            distances_A.append(distance_A)
            
    # for B
    for goal_perm_B in goal_permutations_indices_B:
        for init_perm_B in init_permutations_indices_B:
            # Calculate Euclidean distance for the current permutation
            distance_B = [np.linalg.norm(goal_index - initial_index) for goal_index, initial_index in zip(np.array(goal_perm_B), np.array(init_perm_B))]
            distance_B = sum(distance_B)
            distances_B.append(distance_B)
            
            
    min_distance_A = min(distances_A)
    min_distance_B = min(distances_B)
    
    # Normalize distances by dividing by the maximum distance
    
    
    # count how many elements are not equal to E in the v_matrix
    count = np.count_nonzero(v_matrix == 'A') + np.count_nonzero(v_matrix == 'B')
    #print("Count :", count)

    tot_distance = (min_distance_A  + min_distance_B) / count
    
    return tot_distance


def euclidean_v2(v, goal_v, move1, move2, place1, pickup1, missing, penalties):
    
    tot_distance = 0
        
    v_matrix = preprocess(v)
    goal_matrix = preprocess(goal_v)
    
    max_distance = np.linalg.norm(np.array([0, 0]) - np.array([len(goal_matrix)-1, len(goal_matrix[0])-1]))
    
    if missing == 1: 
        missing_element = None
        
        initial_count_A = np.count_nonzero(v_matrix == 'A')
        initial_count_B = np.count_nonzero(v_matrix == 'B')
        
        goal_count_A = np.count_nonzero(goal_matrix == 'A')
        goal_count_B = np.count_nonzero(goal_matrix == 'B')
        
        if initial_count_A < goal_count_A:
            missing_element = 'A'
        elif initial_count_B < goal_count_B:
            missing_element = 'B'
        
        if missing_element is not None:
            v = v.replace("ra-world-arm", missing_element)
            
        v_matrix = preprocess(v)
        
        tot_distance += euclidean_distance_v2(v_matrix, goal_matrix)
        
        if move1 == 1:
            tot_distance += penalties['move1']
        elif pickup1 == 1:
            tot_distance += penalties['pickup1']
        elif move2 == 1:
            tot_distance += penalties['move2']
    
    else: 
        tot_distance += euclidean_distance_v2(v_matrix, goal_matrix)
        
        if move1 == 1:
            tot_distance += penalties['move1']
        elif place1 == 1:
            tot_distance += penalties['place1']
        elif move2 == 1: 
            tot_distance += penalties['move2']
        
    normalized_distance = tot_distance / max_distance if max_distance != 0 else 0
    
    return normalized_distance

