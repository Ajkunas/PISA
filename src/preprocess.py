import pandas as pd 
from distances import euclidean_v2
import numpy as np
import ast

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

################################## CONSTANTS ####################################################

L1 = "P1M120"
L2 = "P1M123"
L3 = "P1M124"

KEY_VECTOR_L1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,ra-world-shape ra-world-shapeA,', 'E,E,E,E,ra-world-shape ra-world-shapeA,false']"
KEY_VECTOR_L2 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,', 'E,ra-world-shape ra-world-shapeB,E,ra-world-shape ra-world-shapeA,E,false']"
KEY_VECTOR_L3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,E,E,', 'ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,E,E,false']"

INIT_WORLDSPACE_L1 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,', 'ra-world-shape ra-world-shapeA,E,E,E,E,false']"
INIT_WORLDSPACE_L2 = "['E,E,ra-world-arm,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeA,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,false']"
INIT_WORLDSPACE_L3 = "['ra-world-arm,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,E,E,E,', 'E,E,ra-world-shape ra-world-shapeB,E,E,', 'E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeA,E,E,', 'E,ra-world-shape ra-world-shapeA,ra-world-shape ra-world-shapeB,E,E,false']" 

PENALTIES = {'move1': 0.1, 'move2': 0.1, 'pickup1': 0.1, 'place1': 0.1}
NO_PENALTIES = {'move1': 0, 'move2': 0, 'pickup1': 0, 'place1': 0}
OPTIMAL_PENALTIES = {'move1': 0.3, 'move2': 0.3, 'pickup1': 0.3, 'place1': 0.3}

##################################################################################################

def preprocess_world(vector): 
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

# A bad worldspace is a worldspace which has more element than it should
def detect_longer_worldspace(v, goal_v): 
    v_matrix = preprocess_world(v)
    goal_matrix = preprocess_world(goal_v)
    
    more_elements = np.count_nonzero(v_matrix == 'A') > np.count_nonzero(goal_matrix == 'A') or np.count_nonzero(v_matrix == 'B') > np.count_nonzero(goal_matrix == 'B')   
    
    return more_elements

# worldspace which as no element at all (donc 30 E elements)
def detect_missing_worldspace(v, goal_v): 
    v_matrix = preprocess_world(v)
    goal_matrix = preprocess_world(goal_v)
    
    missing_A = np.count_nonzero(goal_matrix == 'A') - np.count_nonzero(v_matrix == 'A')
    missing_B = np.count_nonzero(goal_matrix == 'B') - np.count_nonzero(v_matrix == 'B')
    
    return missing_A > 1 or missing_B > 1

def detect_error(worldspace):
    worldspace = worldspace.replace("ra-world-shape ra-world-shapeA", "A")
    worldspace = worldspace.replace("ra-world-shape ra-world-shapeB", "B")
    worldspace = ast.literal_eval(worldspace)
    
    matrix = []
    for i in range(len(worldspace)):
        matrix.append(worldspace[i].split(','))
        
    return int(matrix[-1][-1] != 'false')


def detect_type_error(worldspace, type_error):
    worldspace = worldspace.replace("ra-world-shape ra-world-shapeA", "A")
    worldspace = worldspace.replace("ra-world-shape ra-world-shapeB", "B")
    worldspace = ast.literal_eval(worldspace)
    
    matrix = []
    for i in range(len(worldspace)):
        matrix.append(worldspace[i].split(','))
    
    
    if matrix[-1][-1] == type_error:
        return 1
    else :
        return 0
    
    
def detect_missing_element(v, goal_v): 
    
    v_matrix = preprocess_world(v)
    goal_matrix = preprocess_world(goal_v)
    
    initial_count_A = np.count_nonzero(v_matrix == 'A')
    initial_count_B = np.count_nonzero(v_matrix == 'B')

    goal_count_A = np.count_nonzero(goal_matrix == 'A')
    goal_count_B = np.count_nonzero(goal_matrix == 'B')
        
    error = initial_count_A != goal_count_A or initial_count_B != goal_count_B
    
    return int(error)

def cleaning_data(df, key_vector): 
    df = df.dropna(subset=['WorldSpace'])
    df = df[~df['WorldSpace'].str.contains("trial")]
    
    df['longer_worldspace'] = df.apply(lambda x: detect_longer_worldspace(x['WorldSpace'], key_vector), axis=1)
    
    # remove rows where bad_worldspace is True
    df = df[~df['longer_worldspace']]
    
    # remove the bad_worldspace column
    df = df.drop(columns=['longer_worldspace'])
    
    df['missing_worldspace'] = df.apply(lambda x: detect_missing_worldspace(x['WorldSpace'], key_vector), axis=1)
    
    # remove rows where missing_worldspace is True
    df = df[~df['missing_worldspace']]
    
    # remove the missing_worldspace column
    df = df.drop(columns=['missing_worldspace'])
    
    df_grouped = df.groupby('Student ID').apply(lambda x: 1 in x['WorldspaceScore'].values)

    # separate the students into two groups
    success_students = df_grouped[df_grouped == True].index
    failure_students = df_grouped[df_grouped == False].index
    
    # Sanity check to verify if the failing students do not overlap over the successful students (or vice-versa)
    if set(success_students.intersection(failure_students)) == set():
        print("Finished cleaning !")
    else: 
        print("Failing and successful students overlap: problem !")
    
    return df

def find_minimum(df, column): 
    data = df[column]

    # Compute the KDE
    kde = gaussian_kde(data)

    # Evaluate the KDE at a range of points
    x_values = np.linspace(data.min(), data.max(), 1000)
    kde_values = kde.evaluate(x_values)

    # Invert the KDE values to find local minima instead of maxima
    inverted_kde_values = -kde_values

    # Find local minima using find_peaks
    minima_indices, _ = find_peaks(inverted_kde_values)

    local_minima = x_values[minima_indices]

    #print("Local minima of the KDE curve:")
    #for minimum in local_minima:
        #print("x =", minimum)
       
    return local_minima[0]

def bucketization(df, type):
    df_l1 = df[df['activity'] == 1]
    df_l2 = df[df['activity'] == 2]
    df_l3 = df[df['activity'] == 3]
    
    dfs = [df_l1, df_l2, df_l3]
    
    if type == 'distribution':
        for col in ['delta_successive', 'Submission_TreeDist_Successive']:
            thresholds = []
            for idx, data in enumerate(dfs):
                min_kde_value = find_minimum(data, col)
                thresholds.append(min_kde_value)

                #print(f"Global minimum of the KDE curve for task {idx} :", min_kde_value)
            for data, thresh in zip(dfs, thresholds):
                data[f'bucket_{col}'] = data[col].apply(lambda x: "low" if x <= thresh else "high")
            
    elif type == 'median':
        for col in ['delta_successive', 'Submission_TreeDist_Successive']:
            thresholds = []
            for idx, data in enumerate(dfs):
                thresholds.append(data[col].median())

                #print(f"Global minimum of the KDE curve for task {idx} :", min_kde_value)
            for data, thresh in zip(dfs, thresholds):
                data[f'median_split_{col}'] = data[col].apply(lambda x: "low" if x <= thresh else "high")
            
    else:
        print("Type not recognized")
    
    return pd.concat(dfs)

def preprocessing_data(df, key_vector, initial_worldspace, activity_id, penalties, code=False): 
    
    # Index the tentatives 
    df = df.groupby('Student ID', as_index=False).apply(lambda x: x.sort_values(by='timestamp')).reset_index(drop=True)
    df['index'] = df.groupby('Student ID').cumcount()
    
    # Selecting only useful informations 
    if code:
        data = df[["Student ID", "WorldSpace", "index", "Submission_TreeDist_Successive"]]
    else: 
        data = df[["Student ID", "WorldSpace", "index"]]
        

    # dictionnary to reassign index to the correct order, where index 0 becomes 1 and index 1 becomes 2
    index_dict = {i: i+1 for i in range(len(data['index'].unique()))}

    data['index'] = data['index'].apply(lambda x: index_dict[x])
    
    # Adding initial vector as starting point 
    
    if code:
        initial_data = pd.DataFrame({'Student ID': data['Student ID'].unique(), 'WorldSpace': initial_worldspace, 'index': 0, "Submission_TreeDist_Successive": 0})
    else: 
        initial_data = pd.DataFrame({'Student ID': data['Student ID'].unique(), 'WorldSpace': initial_worldspace, 'index': 0})
        
    data = pd.concat([data, initial_data])

    data = data.groupby('Student ID', as_index=False).apply(lambda x: x.sort_values(by='index')).reset_index(drop=True)
    
    # create column max tentative which correspond to the last row column "index" for each student
    data["nb_tentative"] = data.groupby('Student ID')['index'].transform('max')
    
    # Create column corresponding to percentage of activity completed 
    data['pct_activity'] = data['index'] / data['nb_tentative']
    
    # Create column to indicate if the student made an error or not during the tentative
    data['error'] = data['WorldSpace'].apply(lambda x: detect_error(x))
    
    data['missing'] = data['WorldSpace'].apply(lambda x: detect_missing_element(x, key_vector))
    
    for error in ['move1', 'move2', 'pickup1', 'place1']:
        data[error] = data['WorldSpace'].apply(lambda x: detect_type_error(x, error))
        
    for x in ['error', 'missing', 'move1', 'move2', 'pickup1', 'place1']:
        data[f"nb_{x}"] = data.groupby('Student ID')[x].transform('sum')
        
    # Add percentage columns 
    for x in ['error', 'missing', 'move1', 'move2', 'pickup1', 'place1']:
        data[f'pct_{x}'] = data[f'nb_{x}'] / data['nb_tentative']
        
    # Compute euclidean distance 
    for idx, row in data.iterrows():
        missing = row['missing']
        move1 = row['move1']
        move2 = row['move2']
        pickup1 = row['pickup1']
        place1 = row['place1']
        
        data.loc[idx, 'euclidean_distance'] = euclidean_v2(row['WorldSpace'], key_vector, move1, move2, 
                                                            place1, pickup1, missing, penalties)
        
    # Compute euclidean distance successively 
    for idx, row in data.iterrows(): 
        if row['index'] == 0: 
            data.loc[idx, 'delta_successive'] = 0
        else: 
            data.loc[idx, 'delta_successive'] = euclidean_v2(data.loc[idx, 'WorldSpace'], data.loc[idx-1, 'WorldSpace'], 
                                            data.loc[idx, 'move1'], data.loc[idx, 'move2'], data.loc[idx, 'place1'], 
                                            data.loc[idx, 'pickup1'], data.loc[idx, 'missing'], penalties)
            
    # Separate successful and failing students 
    df_grouped = df.groupby('Student ID').apply(lambda x: 1 in x['WorldspaceScore'].values)

    success_students = df_grouped[df_grouped == True].index
    failure_students = df_grouped[df_grouped == False].index
    
    data_success = data[data['Student ID'].isin(success_students)]
    data_fail = data[data['Student ID'].isin(failure_students)]
    
    data_success['success'] = 1
    data_fail['success'] = 0
    
    data = pd.concat([data_success, data_fail])
    
    # adding columns for sequence mining
    for idx, row in data.iterrows():
        data.at[idx, 'success_seq'] = 'success' if row['success'] == 1 else 'fail'
    
    # Separate students with an error or not
    error_student = data[data['nb_error'] != 0]['Student ID'].unique()
    no_error_student = data[data['nb_error'] == 0]['Student ID'].unique()
    
    data_error = data[data['Student ID'].isin(error_student)]
    data_no_error = data[data['Student ID'].isin(no_error_student)]
    
    data_error['has_error'] = 1
    data_no_error['has_error'] = 0
    
    data = pd.concat([data_error, data_no_error])
    
    data['activity'] = activity_id
    
    for idx, row in data.iterrows():
        if row['move1'] == 1: 
            data.loc[idx, 'error_seq'] = 'move1'
        elif row['move2'] == 1:
            data.loc[idx, 'error_seq'] = 'move2'
        elif row['pickup1'] == 1:
            data.loc[idx, 'error_seq'] = 'pickup1'
        elif row['place1'] == 1:
            data.loc[idx, 'error_seq'] = 'place1'
        else: 
            data.loc[idx, 'error_seq'] = 'none'
            
    # adding columns for sequence mining
    for idx, row in data.iterrows():
        data.at[idx, 'error_seq_gen'] = 'error' if row['error_seq'] != 'none' else 'none'
            
    data['case1'] = 0
    data['case2'] = 0
    data['case3'] = 0
    
    for idx, row in data.iterrows():
        if row['error'] == 1 and row['missing'] == 1:  
            data.loc[idx, 'cases'] = 'case1'
            data.loc[idx, 'case1'] = 1
        elif row['error'] == 1 and row['missing'] != 1:
            data.loc[idx, 'cases'] = 'case2'
            data.loc[idx, 'case2'] = 1
        elif row['error'] != 1 and row['missing'] == 1:
            data.loc[idx, 'cases'] = 'case3'
            data.loc[idx, 'case3'] = 1
        else:
            data.loc[idx, 'cases'] = 'none'
    
    # adding columns for sequence mining
    for idx, row in data.iterrows():
        data.at[idx, 'case_seq'] = 'case' if row['cases'] != 'none' else 'none'
    
    return data


def preprocessing_general(df, key_vectors, initial_worldspaces, activity_ids, activity_nbs, penalties, code=False):
    data = pd.DataFrame()
    
    for key_vector, initial_worldspace, activity_id, activity_nb in zip(key_vectors, initial_worldspaces, activity_ids, activity_nbs):
        df_activity = df[df['Activity ID'] == activity_id]
        df_cleaned = cleaning_data(df_activity, key_vector)
        data_preprocessed = preprocessing_data(df_cleaned, key_vector, initial_worldspace, activity_nb, penalties
                                            , code=code)
        data = pd.concat([data, data_preprocessed])
        
    student_activity = data.groupby('Student ID').agg({'activity': 'unique'}).reset_index()
    student_activity['activity_all'] = student_activity['activity'].apply(lambda x: ''.join(map(str, x)))
    student_to_activity_dict = dict(zip(student_activity['Student ID'], student_activity['activity_all']))

    data['activity_all'] = data['Student ID'].map(student_to_activity_dict)
    
    # for pattern mining
    data = bucketization(data, 'distribution')
    data = bucketization(data, 'median')
    
    # help to filter
    data['at_least_2_tentatives'] = data['nb_tentative'].apply(lambda x: 1 if x >= 2 else 0)
    
    return data
        


def grouped_per_student(df, code=False): 
    if code:
        data_grouped_per_student = df.groupby('Student ID').agg({'euclidean_distance': 'mean', 'delta_successive': 'mean', "Submission_TreeDist_Successive": 'mean', 'nb_tentative': 'max', 
                                                            'nb_error': 'max', 'nb_move1': 'max', 'nb_move2': 'max', 
                                                            'nb_pickup1': 'max', 'nb_place1': 'max', 'pct_error': 'max', 
                                                            'pct_move1': 'max', 'pct_move2': 'max', 'pct_pickup1': 'max', 
                                                            'pct_place1': 'max', 'success': 'max', 'has_error': 'max', 'activity_all': 'max'}).reset_index()
        data_grouped_per_student.columns = ['Student ID', 'mean_euclidean_distance', 'mean_delta_successive', "mean_code_distance", 'nb_tentative', 'nb_error', 'nb_move1', 
                                        'nb_move2', 'nb_pickup1', 'nb_place1', 'pct_error', 'pct_move1', 
                                        'pct_move2', 'pct_pickup1', 'pct_place1', 'success', 'has_error', 'activity_all']
        return data_grouped_per_student
    else: 
        data_grouped_per_student = df.groupby('Student ID').agg({'euclidean_distance': 'mean', 'delta_successive': 'mean', 'nb_tentative': 'max', 
                                                            'nb_error': 'max', 'nb_move1': 'max', 'nb_move2': 'max', 
                                                            'nb_pickup1': 'max', 'nb_place1': 'max', 'pct_error': 'max', 
                                                            'pct_move1': 'max', 'pct_move2': 'max', 'pct_pickup1': 'max', 
                                                            'pct_place1': 'max', 'success': 'max', 'has_error': 'max', 'activity_all': 'max'}).reset_index()
        data_grouped_per_student.columns = ['Student ID', 'mean_euclidean_distance', 'mean_delta_successive', 'nb_tentative', 'nb_error', 'nb_move1', 
                                        'nb_move2', 'nb_pickup1', 'nb_place1', 'pct_error', 'pct_move1', 
                                        'pct_move2', 'pct_pickup1', 'pct_place1', 'success', 'has_error', 'activity_all'] 

        return data_grouped_per_student