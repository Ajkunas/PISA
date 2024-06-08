import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from models import LogisticRegressionModel, LSTMModel
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

from utils import split_data, pad_with_pattern
from sequence_creator import SequenceCreator

import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, model_type, sequence_creator, epochs=100, lr=0.01, weight_decay=0.01, dropout=0.2, hidden_dim=100, test_size=0.2, task="baseline", evaluate=True, k=5):
        self.model_type = model_type
        self.sequence_creator = sequence_creator
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.test_size = test_size
        self.task = task
        self.evaluate = evaluate
        self.k = k

    def train(self):
        train_size = 1 - self.test_size
        world_seq_train, world_seq_test, world_seq_valid = split_data(self.sequence_creator.world_sequences, train_size)
        code_seq_train, code_seq_test, code_seq_valid = split_data(self.sequence_creator.code_sequences, train_size)
        error_seq_train, error_seq_test, error_seq_valid = split_data(self.sequence_creator.error_sequences, train_size)
        case_seq_train, case_seq_test, case_seq_valid = split_data(self.sequence_creator.case_sequences, train_size)
        success_seq_train, success_seq_test, success_seq_valid = split_data(self.sequence_creator.success_sequences, train_size)

        X_train, y_train = self.sequence_creator.create_features_labels(self.task, world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train)
        X_test, y_test = self.sequence_creator.create_features_labels(self.task, world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test)
        X_valid, y_valid = self.sequence_creator.create_features_labels(self.task, world_seq_valid, code_seq_valid, error_seq_valid, case_seq_valid, success_seq_valid)
        
        #print("X_train", X_train)
        max_len = max(len(seq) for seq in X_train)
        pad_value = [0, 1] if self.sequence_creator.one_hot else [0]
        
        if not self.sequence_creator.one_hot or self.task in ["code_error_n_only", "code_n_only", "error_n_only"]:
            X_train_padded = X_train
            X_test_padded = X_test
            X_valid_padded = X_valid
        else:
            X_train_padded = pad_with_pattern(X_train, max_len, pad_value)
            X_test_padded = pad_with_pattern(X_test, max_len, pad_value)
            X_valid_padded = pad_with_pattern(X_valid, max_len, pad_value)
  

        if self.sequence_creator.one_hot:
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
            y_valid = np.argmax(y_valid, axis=1)
        else:
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            y_valid = np.array(y_valid)
        
        if self.model_type == 'rf':
            model = RandomForestClassifier()
            model.fit(X_train_padded, y_train)
            y_prob = model.predict_proba(X_test_padded)[:, 1]
            y_pred = model.predict(X_test_padded)
            accuracy = accuracy_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            loss_train_arr, loss_valid_arr = [], []
        else:

            X_train = torch.tensor(X_train_padded, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test_padded, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            X_valid = torch.tensor(X_valid_padded, dtype=torch.float32)
            y_valid = torch.tensor(y_valid, dtype=torch.float32)

            if not self.sequence_creator.one_hot and self.task in ["code_error_n_only", "code_n_only", "error_n_only"]:
                X_train = X_train.unsqueeze(-1)
                X_test = X_test.unsqueeze(-1)
                X_valid = X_valid.unsqueeze(-1)

            input_dim = X_train.shape[1]

            model = None

            if self.model_type == 'logistic':
                model = LogisticRegressionModel(input_dim, self.dropout)
            elif self.model_type == 'lstm':
                model = LSTMModel(1, self.hidden_dim, 1, self.dropout)
                X_train = X_train.unsqueeze(-1)
                X_test = X_test.unsqueeze(-1)
                X_valid = X_valid.unsqueeze(-1)
            else:
                raise ValueError("Model not implemented")

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            loss_train_arr, loss_valid_arr = self.train_model(model, X_train, y_train, X_valid, y_valid, criterion, optimizer)

            if self.evaluate:
                roc_auc, accuracy, fpr, tpr = self.evaluate_model(model, X_test, y_test)
            else:
                roc_auc, accuracy, fpr, tpr = 0, 0, 0

        return loss_train_arr, loss_valid_arr, roc_auc, accuracy, fpr, tpr

    def cross_validate(self):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

        auc_scores = []
        acc_scores = []
        results = []
        
        world_sequences = np.array(self.sequence_creator.world_sequences, dtype=object)
        code_sequences = np.array(self.sequence_creator.code_sequences, dtype=object)
        error_sequences = np.array(self.sequence_creator.error_sequences, dtype=object)
        case_sequences = np.array(self.sequence_creator.case_sequences, dtype=object)
        success_sequences = np.array(self.sequence_creator.success_sequences, dtype=object)
        #print("world_sequences : ", world_sequences)
        #print("code_sequences : ", code_sequences)

        for fold, (train_index, test_index) in enumerate(kf.split(self.sequence_creator.world_sequences)):
            
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

            X_train, y_train = self.sequence_creator.create_features_labels(self.task, world_seq_train, code_seq_train, error_seq_train, case_seq_train, success_seq_train)
            X_test, y_test = self.sequence_creator.create_features_labels(self.task, world_seq_test, code_seq_test, error_seq_test, case_seq_test, success_seq_test)
            
            if not self.sequence_creator.one_hot and self.task in ["code_error_n_only", "code_n_only", "error_n_only"]:
                X_train_padded = X_train
                X_test_padded = X_test
            else:
                max_len = max(len(seq) for seq in X_train)
                pad_value = [0, 1] if self.sequence_creator.one_hot else [0]
                X_train_padded = pad_with_pattern(X_train, max_len, pad_value)
                X_test_padded = pad_with_pattern(X_test, max_len, pad_value)

            if self.sequence_creator.one_hot:
                y_train = np.argmax(y_train, axis=1)
                y_test = np.argmax(y_test, axis=1)
            else:
                y_train = np.array(y_train)
                y_test = np.array(y_test)

            X_train = torch.tensor(X_train_padded, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test_padded, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            if not self.sequence_creator.one_hot and self.task in ["code_error_n_only", "code_n_only", "error_n_only"]:
                X_train = X_train.unsqueeze(-1)
                X_test = X_test.unsqueeze(-1)

            input_dim = X_train.shape[1]

            model = None

            if self.model_type == 'logistic':
                model = LogisticRegressionModel(input_dim, self.dropout)
            elif self.model_type == 'lstm':
                model = LSTMModel(1, self.hidden_dim, 1, self.dropout)
                X_train = X_train.unsqueeze(-1)
                X_test = X_test.unsqueeze(-1)
            elif self.model_type == 'rf':
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                acc_scores.append(accuracy)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                continue
            else:
                raise ValueError("Model not implemented")

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            for epoch in range(self.epochs):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, y_train.view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            roc_auc, accuracy, _, _ = self.evaluate_model(model, X_test, y_test)
            auc_scores.append(roc_auc)
            acc_scores.append(accuracy)

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

    def train_model(self, model, X_train, y_train, X_valid, y_valid, criterion, optimizer):
        loss_train_arr, loss_valid_arr = [], []
        for epoch in range(self.epochs):
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
                loss_valid_arr.append(loss_valid.item())
                
        return loss_train_arr, loss_valid_arr

    def evaluate_model(self, model, X, y):
        with torch.no_grad():
            model.eval()
            outputs = model(X)
            predicted = torch.round(outputs)
            accuracy = (predicted == y.view(-1, 1)).sum().item() / len(y)
            fpr, tpr, thresholds = roc_curve(y.numpy(), outputs.numpy())
            roc_auc = auc(fpr, tpr)
        return roc_auc, accuracy, fpr, tpr
