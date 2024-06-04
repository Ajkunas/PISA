import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models import LogisticRegressionModel, LSTMModel

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

from utils import one_hot_encode_sequences, prepare_sequences, split_data, pad_with_pattern

def create_features_labels_corr(world_sequences, code_sequences, error_sequences, case_sequences, task="baseline", one_hot=False):
    X, y = [], []
    
    task_mapping = {}
    
    if one_hot:
        world_sequences, _ = one_hot_encode_sequences(world_sequences)
        code_sequences, _ = one_hot_encode_sequences(code_sequences)
        error_sequences, _ = one_hot_encode_sequences(error_sequences)
        case_sequences, _ = one_hot_encode_sequences(case_sequences)

        task_mapping = {
            "baseline": lambda w, co, e, ca, i: (w[:i].flatten(), w[i]),
            "code_error_n_world_n_prev": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), e[:i+1].flatten(), w[:i].flatten()]), w[i]),
            "code_error_n": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), e[:i+1].flatten()]), w[i]),
            "code_error_n_prev": lambda w, co, e, ca, i: (np.hstack([co[:i].flatten(), e[:i].flatten()]), w[i]),
            "code_error_n_only": lambda w, co, e, ca, i: (np.hstack([co[i].flatten(), e[i].flatten()]), w[i]),
            "code_n_only": lambda w, co, e, ca, i: (co[i].flatten(), w[i]),
            "error_n_only": lambda w, co, e, ca, i: (e[i].flatten(), w[i]),
            "code_n_until": lambda w, co, e, ca, i: (co[:i+1].flatten(), w[i]),
            "code_n_prev_until": lambda w, co, e, ca, i: (co[:i].flatten(), w[i]),
            "code_n_until_error_n": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), e[i].flatten()]), w[i]),
            "code_case_n": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), ca[i].flatten()]), w[i]),
            "code_case_n_until": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), ca[:i+1].flatten()]), w[i]),
            "code_error_n_world_n_prev_only": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), e[:i+1].flatten(), w[i-1].flatten()]), w[i]),
            "code_n_world_n_prev_only": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), w[i-1].flatten()]), w[i]),
        }
        
    else: 
        
        task_mapping = {
            "baseline": lambda w, c, e, cas, i: (w[:i], w[i]),
            "code_error_n_world_n_prev": lambda w, c, e, cas, i: (c[:i+1] + e[:i+1] + w[:i], w[i]),
            "code_error_n": lambda w, c, e, cas, i: (c[:i+1] + e[:i+1], w[i]),
            "code_error_n_prev": lambda w, c, e, cas, i: (c[:i] + e[:i], w[i]),
            "code_error_n_only": lambda w, c, e, cas, i: (c[i] + e[i], w[i]),
            "code_n_only": lambda w, c, e, cas, i: (c[i], w[i]),
            "error_n_only": lambda w, c, e, cas, i: (e[i], w[i]),
            "code_n_until": lambda w, c, e, cas, i: (c[:i+1], w[i]),
            "code_n_prev_until": lambda w, c, e, cas, i: (c[:i], w[i]),
            "code_n_until_error_n": lambda w, c, e, cas, i: (c[:i+1] + [e[i]], w[i]),
            "code_case_n": lambda w, c, e, cas, i: (c[:i+1] + [cas[i]], w[i]),
            "code_case_n_until": lambda w, c, e, cas, i: (c[:i+1] + cas[:i+1], w[i]),
            "code_error_n_world_n_prev_only": lambda w, c, e, cas, i: (c[:i+1] + e[:i+1] + [w[i-1]], w[i]),
            "code_n_world_n_prev_only": lambda w, c, e, cas, i: (c[:i+1] + [w[i-1]], w[i]),
        }

    for world_seq, code_seq, error_seq, case_seq in zip(world_sequences, code_sequences, error_sequences, case_sequences):
        for i in range(1, len(world_seq)):
            if task in task_mapping:
                features, label = task_mapping[task](world_seq, code_seq, error_seq, case_seq, i)
                X.append(features)
                y.append(label)
    
    return X, y



def create_features_labels_success(world_sequences, code_sequences, error_sequences, case_sequences, success_sequences, task="baseline", one_hot=False):
    X, y = [], []
    
    if one_hot:
        world_sequences, _ = one_hot_encode_sequences(world_sequences)
        code_sequences, _ = one_hot_encode_sequences(code_sequences)
        error_sequences, _ = one_hot_encode_sequences(error_sequences)
        case_sequences, _ = one_hot_encode_sequences(case_sequences)
        success_sequences, _ = one_hot_encode_sequences(success_sequences)

        task_mapping = {
            "error": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), e.flatten()]), s[0]),
            "case": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), ca.flatten()]), s[0])
        }
        
    else: 
        
        task_mapping = {
            "error": lambda w, co, e, cas, s: (w + co + e, s[0]),
            "case": lambda w, co, e, ca, s: (w + co + ca, s[0])
        }

    for world_seq, code_seq, error_seq, case_seq, success_seq  in zip(world_sequences, code_sequences, error_sequences, case_sequences, success_sequences):
            if task in task_mapping:
                features, label = task_mapping[task](world_seq, code_seq, error_seq, case_seq, success_seq)
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
          activities=[1,1,1], split_type="distribution", task="baseline", prediction="correlation", evaluate=True, one_hot=False): 
    
    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
    train_size = 1 - test_size
    world_seq_train, world_seq_test, world_seq_valid = split_data(world_sequences, train_size)
    code_seq_train, code_seq_test, code_seq_valid = split_data(code_sequences, train_size)
    error_seq_train, error_seq_test, error_seq_valid = split_data(error_sequences, train_size)
    case_seq_train, case_seq_test, case_seq_valid = split_data(case_sequences, train_size)
    success_seq_train, success_seq_test, success_seq_valid = split_data(success_sequences, train_size)
    

    if prediction == "correlation":
        X_train, y_train = create_features_labels_corr(world_seq_train, code_seq_train, error_seq_train, case_seq_train, task, one_hot)
        X_test, y_test = create_features_labels_corr(world_seq_test, code_seq_test, error_seq_test, case_seq_test, task, one_hot)
        X_valid, y_valid = create_features_labels_corr(world_seq_valid, code_seq_valid, error_seq_valid, case_seq_valid, task, one_hot)
    elif prediction == "success":
        X_train, y_train = create_features_labels_success(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
        X_test, y_test = create_features_labels_success(world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test, task, one_hot)
        X_valid, y_valid = create_features_labels_success(world_seq_valid, code_seq_valid, error_seq_valid, case_seq_valid, success_seq_valid, task, one_hot)
    else: 
        raise ValueError("Prediction not implemented")

    max_len = max(len(seq) for seq in X_train)
    pad_value = [0, 1] if one_hot else [0]
    
    if not one_hot or task in ["code_error_n_only", "code_n_only", "error_n_only"]:
        X_train_padded = X_train
        X_test_padded = X_test
        X_valid_padded = X_valid
        
    else:
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
    
    if not one_hot and task in ["code_error_n_only", "code_n_only", "error_n_only"]:
        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        X_valid = X_valid.unsqueeze(-1)
            
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
                   hidden_dim=100, activities=[1,1,1], split_type="distribution", task="baseline", prediction="correlation", k=5, one_hot=False):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    world_sequences, code_sequences, error_sequences, case_sequences, success_sequences = prepare_sequences(data, activities, split_type)
    
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
        
        if prediction == "correlation":
            X_train, y_train = create_features_labels_corr(world_seq_train, code_seq_train, error_seq_train, case_seq_train, task, one_hot)
            X_test, y_test = create_features_labels_corr(world_seq_test, code_seq_test, error_seq_test, case_seq_test, task, one_hot)
        elif prediction == "success":
            X_train, y_train = create_features_labels_success(world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train, task, one_hot)
            X_test, y_test = create_features_labels_success(world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test, task, one_hot)
        else:
            raise ValueError("Prediction not implemented")
        
        
        if not one_hot and task in ["code_error_n_only", "code_n_only", "error_n_only"]:
            X_train_padded = X_train
            X_test_padded = X_test
        else:
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
        
        if not one_hot and task in ["code_error_n_only", "code_n_only", "error_n_only"]:
            X_train = X_train.unsqueeze(-1)
            X_test = X_test.unsqueeze(-1)
                
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