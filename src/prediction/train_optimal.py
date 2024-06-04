import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models import LogisticRegressionModel, LSTMModel

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

from utils import prepare_sequences, split_data, pad_with_pattern
from train import create_features_labels_success, train_model, evaluate_model


def truncate_sequences(sequences, n, mode='first'):
    truncated_sequences = []
    for seq in sequences:
        if mode == 'first':
            truncated_sequences.append(seq[:n])
        elif mode == 'last':
            truncated_sequences.append(seq[-n:])
        elif mode == 'both':
            truncated_sequences.append(seq[n:len(seq)-n])
        else:
            raise ValueError("Mode not supported. Use 'first' or 'last'.")
    return truncated_sequences


def sliding_window_sequences(sequences, window_size, mode='first'):
    windowed_sequences = []
    for seq in sequences:
        if len(seq) < window_size:
            windowed_sequences.append(seq)
        else:
            if mode == 'first':
                windowed_sequences.append(seq[:window_size])
            elif mode == 'last':
                windowed_sequences.append(seq[-window_size:])
            else:
                raise ValueError("Mode not supported. Use 'first' or 'last'.")
    return windowed_sequences


def train_optimized(model_type, data, epochs=100, lr=0.01, weight_decay=0.01, dropout=0.2, hidden_dim=100, test_size=0.2, 
          activities=[1,1,1], split_type="distribution", task="baseline", evaluate=True, 
          one_hot=False, sequence_method=None, method_param=5, method_mode='first'): 

    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
    # Apply sequence method if specified
    if sequence_method == 'truncate':
        world_sequences = truncate_sequences(world_sequences, method_param, method_mode)
        code_sequences = truncate_sequences(code_sequences, method_param, method_mode)
        error_sequences = truncate_sequences(error_sequences, method_param, method_mode)
        case_sequences = truncate_sequences(case_sequences, method_param, method_mode)
    elif sequence_method == 'sliding_window':
        world_sequences = sliding_window_sequences(world_sequences, method_param)
        code_sequences = sliding_window_sequences(code_sequences, method_param)
        error_sequences = sliding_window_sequences(error_sequences, method_param)
        case_sequences = sliding_window_sequences(case_sequences, method_param)
    
    train_size = 1 - test_size
    world_seq_train, world_seq_test, world_seq_valid = split_data(world_sequences, train_size)
    code_seq_train, code_seq_test, code_seq_valid = split_data(code_sequences, train_size)
    error_seq_train, error_seq_test, error_seq_valid = split_data(error_sequences, train_size)
    case_seq_train, case_seq_test, case_seq_valid = split_data(case_sequences, train_size)
    success_seq_train, success_seq_test, success_seq_valid = split_data(success_sequences, train_size)
    
    X_train, y_train = create_features_labels_success(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
    X_test, y_test = create_features_labels_success(world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test, task, one_hot)
    X_valid, y_valid = create_features_labels_success(world_seq_valid, code_seq_valid, error_seq_valid, case_seq_valid, success_seq_valid, task, one_hot)

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




def cross_validate_optimized(model_type, data, epochs=100, lr=0.01, weight_decay=0.01, dropout=0.2, 
                   hidden_dim=100, activities=[1,1,1], split_type="distribution", task="baseline",
                   sequence_method=None, method_param=5, method_mode='first', k=5, one_hot=False):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
    # Apply sequence method if specified
    if sequence_method == 'truncate':
        world_sequences = truncate_sequences(world_sequences, method_param, method_mode)
        code_sequences = truncate_sequences(code_sequences, method_param, method_mode)
        error_sequences = truncate_sequences(error_sequences, method_param, method_mode)
        case_sequences = truncate_sequences(case_sequences, method_param, method_mode)
    elif sequence_method == 'sliding_window':
        world_sequences = sliding_window_sequences(world_sequences, method_param)
        code_sequences = sliding_window_sequences(code_sequences, method_param)
        error_sequences = sliding_window_sequences(error_sequences, method_param)
        case_sequences = sliding_window_sequences(case_sequences, method_param)
        
    
    world_sequences = np.array(world_sequences, dtype=object)
    code_sequences = np.array(code_sequences, dtype=object)
    error_sequences = np.array(error_sequences, dtype=object)
    case_sequences = np.array(case_sequences, dtype=object)
    success_sequences = np.array(success_sequences, dtype=object)
    

    auc_scores = []
    acc_scores = []
    
    results = []

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(world_sequences)):
        
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
        
        
        X_train, y_train = create_features_labels_success(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
        X_test, y_test = create_features_labels_success(world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test, task, one_hot)

        
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