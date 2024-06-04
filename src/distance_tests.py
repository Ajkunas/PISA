import numpy as np
from distances import *

# Unit tests for the RobotArm acttivity, testing the world space metrics. 

##### Basic tests for the Learning task 1 #####

KEY_VECTOR_V = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

# Initial position
V1 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"

# Cans are next to each other but in the wrong position (next to initial position)
V2 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,E,E,E,false']"

# Cans are shifted to the left, from one position from the initial position
V3 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,false']"

# Cans next to each other, shifted from one position from the initial position
V4 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,E,E,false']"

# Cans are shifted to the left, from one positions from the initial position
V5 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,false']"

# One can in good position, the other is not: shifted to the left by one position from the good one (next to each other)
V6 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,false']"

# Both cans are in good position
V7 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is at the initial position
V8 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is shifted to the right from the initial position
V9 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,ra-world-shape ra-world-shapeA,false']"

# One can is in good position, the other is not: one is shifted to the right from the initial position
V10 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,ra-world-shape ra-world-shapeA,false']"

# Cans are shifted to the left, from two positions from the initial position
V11 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,false']"

# Comparing vectors from initial state to the goal state by moving one can at a time (from left to right)
LIST_VECTORS_1 = [V1, V2, V3, V4, V5, V6, V7]

# Varying the position of one of the can, while the other is in good position
LIST_VECTORS_2 = [V8, V9, V10, V6]

# Varying the shift of both cans 
LIST_VECTORS_3 = [V1, V3, V11, V5, V7]


############ BASIC TESTS FOR CASE WITH 1 CAN AND 1 GLASS #############

KEY_VECTOR_W = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,false']"

### TEST : going toward the goal state step by step ###

# Initial state where can and glass inversed positions and at the other end
W1 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeB,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"

W2 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,E,E,false']"

W3 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeA,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,E,E,false']"

W4 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,E,E,false']"

W5 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,false']"

W6 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,false']"

W7 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,E,', 'E,E,E,ra-world-shape ra-world-shapeB,E,false']"

W8 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeB,ra-world-shape ra-world-shapeA,false']"

W9 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeB,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"

W10 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,false']"

W11 = "['E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeB,false']"

LIST_VECTOR_W = [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11]
    
### TEST : testing the variation of shifts and order towars the goal state ###
# initial state

KEY_VECTOR_W_MATRIX = np.array([['E', 'E', 'E', 'E', 'E'],
                                ['E', 'E', 'E', 'E', 'E'],
                                ['E', 'E', 'E', 'E', 'E'],
                                ['E', 'E', 'E', 'E', 'E'],
                                ['E', 'E', 'E', 'E', 'A'],
                                ['E', 'E', 'E', 'E', 'B']])


W12 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['B', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E']])

W13 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['B', 'E', 'E', 'E', 'E']])

W14 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E'],
                ['E', 'B', 'E', 'E', 'E']])

W15 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'B', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E']])

W16 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'E']])

W17 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E']])

W18 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'A', 'E'],
                ['E', 'E', 'E', 'B', 'E']])

W19 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'B', 'E'],
                ['E', 'E', 'E', 'A', 'E']])

W20 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'E', 'E', 'A']])

W21 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'A'],
                ['E', 'E', 'E', 'E', 'B']])

LIST_VECTOR_W2 = [W12, W13, W14, W15, W16, W17, W18, W19, W20, W21]

############ BASIC TESTS FOR CASE WITH 2 CANS AND 2 GLASSES - L2 task  (TEST = TRUE) #############

KEY_VECTOR_X = np.array([['E', 'E', 'E', 'E', 'E'],
                            ['E', 'E', 'E', 'E', 'E'],
                            ['E', 'E', 'E', 'E', 'E'],
                            ['E', 'E', 'E', 'E', 'E'],
                            ['E', 'B', 'E', 'A', 'E'],
                            ['E', 'B', 'E', 'A', 'E']])
# Initial state 
X1 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'E']])

## Cases where the student does everything right

X2 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'B', 'B', 'E', 'E']])

X3 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'B', 'B', 'A', 'E']])

X4 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'A', 'E'],
                ['E', 'B', 'B', 'A', 'E']])

X5 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'B', 'E', 'A', 'E'],
                ['E', 'B', 'E', 'A', 'E']])

LIST_VECTOR_X1 = [X1, X2, X3, X4, X5]

## Testing cases where the student makes mistakes

X6 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'B']])

X7 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'B', 'B', 'E']])

X8 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['B', 'E', 'B', 'E', 'E']])

X9 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'E', 'A', 'E', 'E'],
                ['E', 'B', 'B', 'E', 'E']])

LIST_VECTOR_X2 = [X6, X7, X8, X9]


############ BASIC TESTS FOR CASE WITH 2 CANS AND 2 GLASSES  (TEST = TRUE) #############

KEY_VECTOR_Z = np.array([['E', 'E', 'E', 'E', 'E'],
                        ['E', 'E', 'E', 'E', 'E'],
                        ['E', 'E', 'E', 'E', 'E'],
                        ['E', 'E', 'E', 'E', 'A'],
                        ['E', 'E', 'E', 'E', 'B'],
                        ['E', 'E', 'E', 'B', 'A']])
# Initial state 
Z1 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'E', 'A', 'A']])

Z2 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'A', 'B'],
                ['E', 'E', 'E', 'B', 'A']])

Z3 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'A', 'B', 'A']])

Z4 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'B', 'B'],
                ['E', 'E', 'E', 'A', 'A']])

Z5 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'E', 'E', 'A'],
                ['E', 'E', 'E', 'A', 'B']])

Z6 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'B'],
                ['E', 'E', 'E', 'E', 'A'],
                ['E', 'E', 'E', 'B', 'A']])

LIST_VECTOR_Z1 = [Z1, Z2, Z3, Z4, Z5, Z6]

############ BASIC TESTS FOR CASE WITH 3 CANS AND 2 GLASSES - Learning task 3 (TEST = TRUE) #############

KEY_VECTOR_A = np.array([['E', 'E', 'E', 'E', 'E'],
                        ['E', 'E', 'E', 'E', 'E'],
                        ['E', 'E', 'E', 'E', 'E'],
                        ['A', 'E', 'E', 'E', 'E'],
                        ['A', 'B', 'E', 'E', 'E'],
                        ['A', 'B', 'E', 'E', 'E']])

# Towards the result

# Initial state 
A1 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'B', 'E', 'E'],
                ['E', 'A', 'A', 'E', 'E'],
                ['E', 'A', 'B', 'E', 'E']])

A2 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'A', 'E', 'E'],
                ['E', 'A', 'B', 'B', 'E']])

A3 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E'],
                ['A', 'A', 'B', 'B', 'E']])

A4 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'A', 'B', 'B', 'E']])

A5 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'B', 'B', 'E']])

A6 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'B', 'E', 'B', 'E']])

A7 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['A', 'E', 'E', 'E', 'E'],
                ['A', 'B', 'E', 'E', 'E'],
                ['A', 'B', 'E', 'E', 'E']])

LIST_VECTOR_A1 = [A1, A2, A3, A4, A5, A6, A7]

# Varying the results

A8 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'B', 'E', 'E', 'E'],
                ['E', 'A', 'A', 'E', 'E'],
                ['E', 'A', 'B', 'E', 'E']])

A9 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'A', 'E', 'E'],
                ['B', 'A', 'B', 'E', 'E']])

A10 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E'],
                ['E', 'A', 'B', 'B', 'E']])

A11 = np.array([['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E'],
                ['E', 'A', 'E', 'E', 'E'],
                ['E', 'A', 'B', 'E', 'E'],
                ['E', 'A', 'B', 'E', 'E']])

LIST_VECTOR_A2 = [A2, A8, A9, A10, A11]
    
    
########### COMPARING THE DISTANCES ###########

### For Levenshtein distance ###
def test_levenshtein(metric_type, list_vec, goal_vec, test=False, print_results=False):
    if print_results:
        print("Levenshtein distance for", metric_type)
    results = []
    for v in list_vec:
        distance = levenshtein_distance_combined(v, goal_vec, metric_type, test=test)
        results.append(distance)
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Ratcliff distance ###
def test_ratcliff(metric_type, list_vec, goal_vec, test=False, print_results=False):
    if print_results:
        print("Ratcliff distance for", metric_type)
    results = []
    for v in list_vec:
        distance = ratcliff_obershelp_distance_combined(v, goal_vec, metric_type, test=test)
        results.append(distance)
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Jaro distance ###

def test_jaro(metric_type, list_vec, goal_vec, test=False, print_results=False):
    if print_results:
        print("Jaro distance for", metric_type)
    results = []
    for v in list_vec:
        distance = jaro_combined(v, goal_vec, method=metric_type, test=test)
        results.append(distance)
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

### For Jaro-Winkler distance ###
def test_jaro_winkler(metric_type, list_vec, goal_vec, test=False, print_results=False):
    if print_results:
        print("Jaro-Winkler distance for", metric_type)
    results = []
    for v in list_vec:
        distance = jaro_combined(v, goal_vec, similarity_metric='jaro_winkler', method=metric_type, test=test)
        results.append(distance)
        if print_results:
            print("Vector:", v)
            print(distance)
    return results
            
### For Euclidean distance ###
def test_euclidean(list_vec, goal_vec, test=False, print_results=False):
    if print_results:
        print("Euclidean distance")
    results = []
    for v in list_vec:
        distance = euclidean(v, goal_vec, test=test)
        results.append(distance)
        if print_results:
            print("Vector:", v)
            print(distance)
    return results

