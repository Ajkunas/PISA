from metric_tests import *
from distances import *

import warnings
warnings.filterwarnings('ignore')


# main function
def run_distance_tests():
    
    ## Basic tests for sanity check
    
    # Euclidean distance
    NO_PENALTIES = {'move1': 0, 'move2': 0, 'pickup1': 0, 'place1': 0}

    distance = euclidean_v2(KEY_VECTOR_V, KEY_VECTOR_V, 0, 0, 0, 0, 0, NO_PENALTIES)
    print("Distance between V and V: ", distance)
    
    distance = euclidean_v2(KEY_VECTOR_W, KEY_VECTOR_W, 0, 0, 0, 0, 0, NO_PENALTIES)
    print("Distance between W and W: ", distance)
    
    distance = euclidean_v2(KEY_VECTOR_X, KEY_VECTOR_X, 0, 0, 0, 0, 0, NO_PENALTIES)
    print("Distance between V and V: ", distance)
    
    distance = euclidean_v2(KEY_VECTOR_Z, KEY_VECTOR_Z, 0, 0, 0, 0, 0, NO_PENALTIES)
    print("Distance between V and V: ", distance)
    
    # Levendstein distance
    distance = levenshtein_distance(KEY_VECTOR_V, KEY_VECTOR_V)
    print("Levenshtein distance between V and V: ", distance)

    distance = levenshtein_distance(KEY_VECTOR_W, KEY_VECTOR_W)
    print("Levenshtein distance between W and W: ", distance)

    distance = levenshtein_distance(KEY_VECTOR_X, KEY_VECTOR_X)
    print("Levenshtein distance between X and X: ", distance)

    distance = levenshtein_distance(KEY_VECTOR_Z, KEY_VECTOR_Z)
    print("Levenshtein distance between Z and Z: ", distance)
    
    # Ratcliff distance
    distance = ratcliff_obershelp_distance_combined(KEY_VECTOR_V, KEY_VECTOR_V)
    print("Ratcliff Obershelp distance between V and V: ", distance)

    distance = ratcliff_obershelp_distance_combined(KEY_VECTOR_W, KEY_VECTOR_W)
    print("Ratcliff Obershelp distance between W and W: ", distance)

    distance = ratcliff_obershelp_distance_combined(KEY_VECTOR_X, KEY_VECTOR_X)
    print("Ratcliff Obershelp distance between X and X: ", distance)

    distance = ratcliff_obershelp_distance_combined(KEY_VECTOR_Z, KEY_VECTOR_Z)
    print("Ratcliff Obershelp distance between Z and Z: ", distance)
    
    # Jaro distance
    distance = jaro_combined(KEY_VECTOR_V, KEY_VECTOR_V)
    print("Jaro distance between V and V: ", distance)

    distance = jaro_combined(KEY_VECTOR_W, KEY_VECTOR_W)
    print("Jaro distance between W and W: ", distance)

    distance = jaro_combined(KEY_VECTOR_X, KEY_VECTOR_X)
    print("Jaro distance between X and X: ", distance)

    distance = jaro_combined(KEY_VECTOR_Z, KEY_VECTOR_Z)
    print("Jaro distance between Z and Z: ", distance)
    
    ## Test for sequnences of movements
    folder_path = '../plot/plot_tests/'
    
    test_comparing_metrics(LIST_VECTORS_1, KEY_VECTOR_V, 
                       "Comparing vectors from initial state to the goal state by moving one can at a time (from left to right)", 
                       folder_path, nb=1)
    test_comparing_euclidean_jaro(LIST_VECTORS_1, KEY_VECTOR_V, 
                              "Comparing vectors from initial state to the goal state by moving one can at a time (from left to right)", 
                              folder_path, nb=1)
    
    test_comparing_metrics(LIST_VECTORS_2, KEY_VECTOR_V, 
                       "Varying the position of one of the can, while the other is in good position", 
                       folder_path, nb=2)
    
    test_comparing_metrics(LIST_VECTORS_3, KEY_VECTOR_V, "Varying the shift of both cans", 
                       folder_path, nb=3)
    
    test_comparing_metrics(LIST_VECTOR_W, KEY_VECTOR_W, 
                       "With 1 can and 1 glass", folder_path, nb=4)
    test_comparing_euclidean_jaro(LIST_VECTOR_W, KEY_VECTOR_W, 
                              "With 1 can and 1 glass", folder_path, nb=4)
    
    test_comparing_metrics(LIST_VECTOR_W2, KEY_VECTOR_W,
                       "With 1 can and 1 glass: comparing variations of shifting and ordering", 
                       folder_path, nb=5)
    test_comparing_euclidean_jaro(LIST_VECTOR_W2, KEY_VECTOR_W, 
                              "With 1 can and 1 glass: comparing variations of shifting and ordering",
                              folder_path, nb=5)
    
    test_comparing_metrics(LIST_VECTOR_X1, KEY_VECTOR_X, "2 cans and 2 glasses - when student does everything right",
                       folder_path, nb=6)
    test_comparing_euclidean_jaro(LIST_VECTOR_X1, KEY_VECTOR_X, 
                              "2 cans and 2 glasses - when student does everything right",
                              folder_path, nb=6)
    
    test_comparing_metrics(LIST_VECTOR_X2, KEY_VECTOR_X, 
                       "2 cans and 2 glasses - when student makes mistakes with first move (comparing different mistakes)", 
                       folder_path, nb=7)
    test_comparing_euclidean_jaro(LIST_VECTOR_X2, KEY_VECTOR_X, 
                              "2 cans and 2 glasses - when student makes mistakes with first move (comparing different mistakes)", 
                              folder_path, nb=7)
    
    test_comparing_metrics(LIST_VECTOR_Z1, KEY_VECTOR_Z, 
                       "2 cans and 2 glasses - variation of the results",
                       folder_path, nb=8)
    test_comparing_euclidean_jaro(LIST_VECTOR_Z1, KEY_VECTOR_Z, "2 cans and 2 glasses - variation of the results",
                              folder_path, nb=8)
    
    ## Testing the euclidean distance with penalties
    KEY_VECTOR_V_MISSING = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,E,false']"
    PENALTIES = {'move1': 0.1, 'move2': 0.1, 'pickup1': 0.1, 'place1': 0.1}
    
    # Baseline
    distance = euclidean_v2(KEY_VECTOR_V, KEY_VECTOR_V, 0, 0, 0, 0, 0, NO_PENALTIES)
    print("Distance between V and V: ", distance)
    
    # Missing element 
    distance = euclidean_v2(KEY_VECTOR_V_MISSING, KEY_VECTOR_V, 0, 0, 0, 0, 1, PENALTIES)
    print("Distance between V and V with missing element: ", distance)
    
    # Error
    distance = euclidean_v2(KEY_VECTOR_V, KEY_VECTOR_V, 1, 0, 0, 0, 0, PENALTIES)
    print("Distance between V and V with an error: ", distance)
    
    # Error and missing element
    distance = euclidean_v2(KEY_VECTOR_V_MISSING, KEY_VECTOR_V, 1, 0, 0, 0, 1, PENALTIES)
    print("Distance between V and V with an error and a missing element: ", distance)
    
    # Other type of error
    distance = euclidean_v2(KEY_VECTOR_V_MISSING, KEY_VECTOR_V, 0, 1, 0, 0, 1, PENALTIES)
    print("Distance between V and V with an error: ", distance)
    
    
if __name__ == "__main__":
    
    run_distance_tests()
    
    print("Finished running the tests.")