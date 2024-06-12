import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def create_sequences(df, split_type, one_hot=False): 
    # Try to change the values of the columns to numerical values
    if not one_hot:
        change_to_num = {'low': 0, 'high': 1}
        case_change_to_num = {'none': 0, 'case': 1}

        for col in ['bucket_delta_successive', 'bucket_Submission_TreeDist_Successive', 'median_split_delta_successive',
                    'median_split_Submission_TreeDist_Successive', 'case_seq']:
            if col != 'case_seq':
                df[col] = df[col].map(change_to_num)
            else: 
                df[col] = df[col].map(case_change_to_num)
    
    df_student = df.groupby("Student ID")
    world_seq = []
    code_seq = []
    error_seq = []
    case_seq = []
    success_seq = []

    for student, data in df_student:
        
        student_world_seq = []
        student_code_seq = []
        student_error_seq = []
        student_case_seq = []
        student_success_seq = []
        
        for idx, row in data.iterrows():
            
            if split_type == 'distribution':
                student_world_seq.append(row['bucket_delta_successive'])
                student_code_seq.append(row['bucket_Submission_TreeDist_Successive'])
            elif split_type == 'median':
                student_world_seq.append(row['median_split_delta_successive'])
                student_code_seq.append(row['median_split_Submission_TreeDist_Successive'])
                
            if one_hot:
                student_error_seq.append(row['error_seq_gen'])
                student_success_seq.append(row['success_seq'])
            else:
                student_error_seq.append(row['error'])
                student_success_seq.append(row['success'])
            
            student_case_seq.append(row['case_seq'])
            
        world_seq.append(student_world_seq)
        code_seq.append(student_code_seq)
        error_seq.append(student_error_seq)
        case_seq.append(student_case_seq)
        success_seq.append(student_success_seq)
        
    return world_seq, code_seq, error_seq, case_seq, success_seq


def prepare_sequences(df, activities, split_type, one_hot=False): 
    # Filter out the first row and the first attempt
    df = df[df['index'] != 0]
    df = df[df['nb_tentative'] >= 2]
    
    df_l1 = df[df['activity'] == 1]
    df_l2 = df[df['activity'] == 2]
    df_l3 = df[df['activity'] == 3]
    
    world_sequences = []
    code_sequences = []
    error_sequences = []
    case_sequences = []
    success_sequences = []
    
    if activities[0] == 1: 
        world_seq1, code_seq1, error_seq1, case_seq1, success_seq1 = create_sequences(df_l1, split_type, one_hot)
        world_sequences += world_seq1
        code_sequences += code_seq1
        error_sequences += error_seq1
        case_sequences += case_seq1
        success_sequences += success_seq1
        
    if activities[1] == 1:
        world_seq2, code_seq2, error_seq2, case_seq2, success_seq2 = create_sequences(df_l2, split_type, one_hot)
        world_sequences += world_seq2
        code_sequences += code_seq2
        error_sequences += error_seq2
        case_sequences += case_seq2
        success_sequences += success_seq2
        
    if activities[2] == 1:
        world_seq3, code_seq3, error_seq3, case_seq3, success_seq3 = create_sequences(df_l3, split_type, one_hot)
        world_sequences += world_seq3
        code_sequences += code_seq3
        error_sequences += error_seq3
        case_sequences += case_seq3
        success_sequences += success_seq3
        
    return world_sequences, code_sequences, error_sequences, case_sequences, success_sequences

def one_hot_encode_sequences(sequences):
        flat_sequences = [item for sublist in sequences for item in sublist]
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(np.array(flat_sequences).reshape(-1, 1))
        encoded_sequences = [encoder.transform(np.array(seq).reshape(-1, 1)) for seq in sequences]
        return encoded_sequences, encoder


def pad_with_pattern(sequences, pattern, max_length, padding_position='pre'):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padding_length = max_length - len(seq)
            padding = np.tile(pattern, (padding_length // len(pattern)) + 1)[:padding_length]
            if padding_position == 'pre':
                padded_seq = np.concatenate([padding, seq])
            else:
                padded_seq = np.concatenate([seq, padding])
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences, dtype=np.float32)


def pad_sequences(sequences, pad_value, max_len, padding='pre'):
    if padding == 'post':
        #print("Padding sequences in post ! ")
        #print("Padding sequences in pre ! ")
        #print("Sequences before padding: ", len(sequences))
        #print("Max length: ", max_len)
        #print("Sequences before padding: ", sequences)
        temp = [seq + pad_value*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
        #print("Sequences after padding: ", temp)
        return [seq + pad_value*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
    else:
        #print("Padding sequences in pre ! ")
        #print("Padding sequences in pre ! ")
        #print("Sequences before padding: ", len(sequences))
        #print("Max length: ", max_len)
        #print("Sequences before padding: ", sequences)
        temp = [pad_value * (max_len - len(seq)) + seq if len(seq) < max_len else seq[:max_len] for seq in sequences]
        #print("Sequences after padding: ", temp)
        return [pad_value * (max_len - len(seq)) + seq if len(seq) < max_len else seq[:max_len] for seq in sequences]


def split_data(data, train_size, val_ratio=0.5):
        size = len(data)
        train_idx = int(train_size * size)
        val_idx = int(val_ratio * (size - train_idx))

        return (
            data[:train_idx],
            data[train_idx:train_idx + val_idx],
            data[train_idx + val_idx:]
        )
        
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
        #plt.show()
    
    # Plot loss
    plt.plot(loss_train, label='Train')
    plt.plot(loss_valid, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    

def plot_truncate(method, method_params, mean_aucs, std_aucs, baseline_auc, path_to_save):
    
    baseline_auc = float(baseline_auc)
    
    mean_aucs_last = mean_aucs[method]["last"]
    std_aucs_last = std_aucs[method]["last"]
    mean_aucs_first = mean_aucs[method]["first"]
    std_aucs_first = std_aucs[method]["first"]

    # Ensure all values are numeric
    mean_aucs_last = [float(x) for x in mean_aucs_last]
    mean_aucs_first = [float(x) for x in mean_aucs_first]
    std_aucs_last = [float(x) for x in std_aucs_last]
    std_aucs_first = [float(x) for x in std_aucs_first]

    plt.figure(figsize=(8, 4))

    # Plot mean AUCs with error bars
    plt.errorbar(method_params, mean_aucs_last, yerr=std_aucs_last, fmt='-o', label="Last")
    plt.errorbar(method_params, mean_aucs_first, yerr=std_aucs_first, fmt='-s', label="First")

    # Plot the baseline mean AUC line
    plt.axhline(y=baseline_auc, color='r', linestyle='--', label=f"Baseline Mean AUC: {baseline_auc:.2f}")

    # Adding labels and title
    plt.xlabel("Number of Attempts")
    plt.ylabel("Mean AUC")
    plt.title("Mean AUC vs Number of Attempts")
    plt.legend()
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Display the plot
    plt.savefig(path_to_save + f"mean_auc_{method}.png")
    #plt.show()