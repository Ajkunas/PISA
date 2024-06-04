import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

from train import pad_with_pattern, split_data, one_hot_encode_sequences
from models import LogisticRegressionModel, LSTMModel

import matplotlib.pyplot as plt


def create_sequences(df, split_type, one_hot=False): 
    if not one_hot:
        change_to_num = {'low': 0, 'high': 1}
        case_change_to_num = {'none': 0, 'case': 1}

        for col in ['bucket_delta_successive', 'bucket_Submission_TreeDist_Successive', 
                    'median_split_delta_successive', 'median_split_Submission_TreeDist_Successive', 'case_sequence']:
            
            if col != 'case_sequence':
                df[col] = df[col].map(change_to_num)
            else: 
                df[col] = df[col].map(case_change_to_num)
    
    df_student = df.groupby("Student ID")
    
    world_seq = []
    code_seq = []
    error_seq = []
    case_seq = []    
    success_seq = []

    for _, data in df_student:
        student_world_seq = []
        student_code_seq = []
        student_error_seq = []
        student_case_seq = []
        
        for _, row in data.iterrows(): 
            if split_type == 'distribution':
                student_world_seq.append(row['bucket_delta_successive'])
                student_code_seq.append(row['bucket_Submission_TreeDist_Successive'])
            elif split_type == 'median':
                student_world_seq.append(row['median_split_delta_successive'])
                student_code_seq.append(row['median_split_Submission_TreeDist_Successive'])
                
            if one_hot:
                student_error_seq.append(row['error_sequence'])
            else:
                student_error_seq.append(row['error'])
            
            student_case_seq.append(row['case_sequence'])
                
        world_seq.append(student_world_seq)
        code_seq.append(student_code_seq)
        error_seq.append(student_error_seq)
        case_seq.append(student_case_seq)
        
        if one_hot: 
            success_seq.append(data['success_seq'].values[0])
        else: 
            success_seq.append(data['success'].values[0])
        
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


def create_features_labels(world_sequences, code_sequences, error_sequences, case_sequences, success_sequences, task="baseline", one_hot=False):
    X, y = [], []
    
    if one_hot:
        world_sequences, _ = one_hot_encode_sequences(world_sequences)
        code_sequences, _ = one_hot_encode_sequences(code_sequences)
        error_sequences, _ = one_hot_encode_sequences(error_sequences)
        case_sequences, _ = one_hot_encode_sequences(case_sequences)
        success_sequences, _ = one_hot_encode_sequences(success_sequences)

        task_mapping = {
            "error": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), e.flatten()]), s),
            "case": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), ca.flatten()]), s)
        }
        
    else: 
        
        task_mapping = {
            "error": lambda w, co, e, cas, s: (co + w + e, s),
            "error": lambda w, co, e, ca, s: (co + w + ca, s)
        }

    for world_seq, code_seq, error_seq, case_seq, success  in zip(world_sequences, code_sequences, error_sequences, case_sequences, success_sequences):
            if task in task_mapping:
                features, label = task_mapping[task](world_seq, code_seq, error_seq, case_seq, success)
                X.append(features)
                y.append(label)
    return X, y

        
def evaluate_model(model, X, y):
    with torch.no_grad():
        model.eval()
        outputs = model(X)
        predicted = torch.round(outputs)
        accuracy = (predicted == y.view(-1, 1)).sum().item() / len(y)
        fpr, tpr, thresholds = roc_curve(y.numpy(), outputs.numpy())
        roc_auc = auc(fpr, tpr)
        #print(f'Test Accuracy: {accuracy * 100:.2f}%')
        #print(f'ROC AUC: {roc_auc:.4f}')
    return roc_auc, accuracy, fpr, tpr

def train_model(model, X_train, y_train, X_valid, y_valid, criterion, optimizer, epochs):
    loss_train_arr, loss_valid_arr, accuracy_valid_arr = [], [], []
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train_arr.append(loss.item())
        
        with torch.no_grad():
            model.eval()
            outputs = model(X_valid)
            loss_valid = criterion(outputs, y_valid.view(-1, 1))
            preds_valid = torch.round(outputs)
            accuracy_valid = (preds_valid == y_valid.view(-1, 1)).sum().item() / len(y_valid)
            loss_valid_arr.append(loss_valid.item())
            accuracy_valid_arr.append(accuracy_valid)
            
            #if (epoch + 1) % 10 == 0:
               # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {loss_valid.item():.4f}, Validation Accuracy: {accuracy_valid:.4f}')
    
    avg_accuracy_valid = np.mean(accuracy_valid_arr)
    print(f'Average Validation Accuracy: {avg_accuracy_valid:.4f}')
    
    return loss_train_arr, loss_valid_arr


def train(model_type, data, epochs=100, lr=0.01, weight_decay=0.01, dropout=0.2, hidden_dim=100, test_size=0.2, 
          activities=[1,1,1], split_type="distribution", task="baseline", evaluate=True, one_hot=False): 
    
    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
    train_size = 1 - test_size
    world_seq_train, world_seq_test, world_seq_valid = split_data(world_sequences, train_size)
    code_seq_train, code_seq_test, code_seq_valid = split_data(code_sequences, train_size)
    error_seq_train, error_seq_test, error_seq_valid = split_data(error_sequences, train_size)
    case_seq_train, case_seq_test, case_seq_valid = split_data(case_sequences, train_size)
    success_seq_train, success_seq_test, success_seq_valid = split_data(success_sequences, train_size)
    
    X_train, y_train = create_features_labels(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
    X_test, y_test = create_features_labels(world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test, task, one_hot)
    X_valid, y_valid = create_features_labels(world_seq_valid, code_seq_valid, error_seq_valid, case_seq_valid, success_seq_valid, task, one_hot)

    max_len = max(len(seq) for seq in X_train)
    pad_value = [0, 1] if one_hot else [0]
    
    X_train_padded = pad_with_pattern(X_train, max_len, pad_value)
    X_test_padded = pad_with_pattern(X_test, max_len, pad_value)
    X_valid_padded = pad_with_pattern(X_valid, max_len, pad_value)
        
    if one_hot:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
    else: 
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_valid = np.array(y_valid)
    
    X_train = torch.tensor(X_train_padded, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test_padded, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    X_valid = torch.tensor(X_valid_padded, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
            
    input_dim = X_train.shape[1]
    
    model = None
    
    if model_type == 'logistic':
        model = LogisticRegressionModel(input_dim, dropout)
    elif model_type == 'lstm':
        model = LSTMModel(1, hidden_dim, 1, dropout)
        
        X_train = X_train.unsqueeze(-1)  # Add feature dimension
        X_test = X_test.unsqueeze(-1)  # Add feature dimension
        X_valid = X_valid.unsqueeze(-1)  # Add feature dimension
    else:
        raise ValueError("Model not implemented")

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_train_arr, loss_valid_arr = train_model(model, X_train, y_train, X_valid, y_valid, criterion, optimizer, epochs)
    
    if evaluate:
        roc_auc, accuracy, fpr, tpr = evaluate_model(model, X_test, y_test)
    else:
        roc_auc, accuracy, fpr, tpr = 0, 0, 0
            
    return loss_train_arr, loss_valid_arr, roc_auc, accuracy, fpr, tpr


def cross_validate(model_type, data, epochs=100, lr=0.01, weight_decay=0.01, dropout=0.2, 
                   hidden_dim=100, activities=[1,1,1], split_type="distribution", task="baseline", k=5, one_hot=False):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
    world_sequences = np.array(world_sequences, dtype=object)
    code_sequences = np.array(code_sequences, dtype=object)
    error_sequences = np.array(error_sequences, dtype=object)
    case_sequences = np.array(case_sequences, dtype=object)

    auc_scores = []
    acc_scores = []
    
    results = []

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(world_sequences)):
        #print(f'Fold {fold + 1}/{k}')
        
        world_seq_train = world_sequences[train_index]
        code_seq_train = code_sequences[train_index]
        error_seq_train = error_sequences[train_index]
        case_seq_train = case_sequences[train_index]
        success_seq_train = success_sequences[train_index]
        
        world_seq_test = world_sequences[test_index]
        code_seq_test = code_sequences[test_index]
        error_seq_test = error_sequences[test_index]
        case_seq_test = case_sequences[test_index]
        success_seq_test = success_sequences[test_index]
        
        X_train, y_train = create_features_labels(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
        X_test, y_test = create_features_labels(world_seq_test, code_seq_test, error_seq_test, case_seq_test, task, success_seq_test, one_hot)
        
        max_len = max(len(seq) for seq in X_train)
        pad_value = [0, 1] if one_hot else [0]
    
        X_train_padded = pad_with_pattern(X_train, max_len, pad_value)
        X_test_padded = pad_with_pattern(X_test, max_len, pad_value)
            
        if one_hot:
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
        else: 
            y_train = np.array(y_train)
            y_test = np.array(y_test)
        
        X_train = torch.tensor(X_train_padded, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test_padded, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
                
        input_dim = X_train.shape[1]
        
        # Initialize model, loss function, and optimizer
        model = None
        
        if model_type == 'logistic':
            model = LogisticRegressionModel(input_dim, dropout)
            
        elif model_type == 'lstm':
            model = LSTMModel(1, hidden_dim, 1, dropout)
            
            X_train = X_train.unsqueeze(-1)
            X_test = X_test.unsqueeze(-1)
            
        elif model_type == 'rf':
            rf = RandomForestClassifier()
            
            rf.fit(X_train, y_train)
            y_prob = rf.predict_proba(X_test)[:, 1]
            y_pred = rf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            acc_scores.append(accuracy)
            
            # Compute ROC curve and AUC score
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            
            #results.append({
            #    "Fold": fold + 1,
            #    "Test Accuracy": f"{acc_scores[-1] * 100:.2f}%",
            #    "ROC AUC": f"{roc_auc:.4f}"
            #})
            
            continue
            
        else: 
            raise ValueError("Model not implemented")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train the model
        for epoch in range(epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.view(-1, 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        roc_auc, accuracy, _, _ = evaluate_model(model, X_test, y_test)
        auc_scores.append(roc_auc)
        acc_scores.append(accuracy)
        
       # results.append({
        #    "Fold": fold + 1,
        #    "Test Accuracy": f"{acc_scores[-1] * 100:.2f}%",
        #    "ROC AUC": f"{roc_auc:.4f}"
        #})
    
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    results.append({
        "Mean AUC": f"{mean_auc:.2f}",
        "Standard Deviation of AUC": f"{std_auc:.2f}",
        "Mean Accuracy": f"{mean_acc:.2f}",
        "Standard Deviation of Accuracy": f"{std_acc:.2f}"
    })

    return results

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