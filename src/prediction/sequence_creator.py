import numpy as np

from utils import one_hot_encode_sequences, prepare_sequences

class SequenceCreator:
    def __init__(self, data, prediction="correlation", activities=[1, 1, 1], split_type="distribution", one_hot=False, sequence_method=None, method_param=5, method_mode='first'):
        self.data = data
        self.prediction = prediction
        self.activities = activities
        self.split_type = split_type
        self.one_hot = one_hot
        self.sequence_method = sequence_method
        self.method_param = method_param
        self.method_mode = method_mode
        self.world_sequences, self.code_sequences, self.error_sequences, self.case_sequences, self.success_sequences = prepare_sequences(data, activities, split_type)
        self.apply_sequence_method()

    def apply_sequence_method(self):
        if self.sequence_method == 'truncate':
            self.world_sequences = self.truncate_sequences(self.world_sequences, self.method_param, self.method_mode)
            self.code_sequences = self.truncate_sequences(self.code_sequences, self.method_param, self.method_mode)
            self.error_sequences = self.truncate_sequences(self.error_sequences, self.method_param, self.method_mode)
            self.case_sequences = self.truncate_sequences(self.case_sequences, self.method_param, self.method_mode)
        elif self.sequence_method == 'truncate_subsequences':
            self.world_sequences = self.truncate_sequences_subsequences(self.world_sequences, self.method_param, self.method_mode)
            self.code_sequences = self.truncate_sequences_subsequences(self.code_sequences, self.method_param, self.method_mode)
            self.error_sequences = self.truncate_sequences_subsequences(self.error_sequences, self.method_param, self.method_mode)
            self.case_sequences = self.truncate_sequences_subsequences(self.case_sequences, self.method_param, self.method_mode)

    @staticmethod
    def truncate_sequences(sequences, n, mode='first'):
        truncated_sequences = []
        for seq in sequences:
            if mode == 'first':
                truncated_sequences.append(seq[n:])
            elif mode == 'last':
                truncated_sequences.append(seq[:len(seq)-n])
            elif mode == 'both':
                truncated_sequences.append(seq[n:len(seq)-n])
            else:
                raise ValueError("Mode not supported. Use 'first', 'last' or 'both'.")
        return truncated_sequences

    @staticmethod
    def truncate_sequences_subsequences(sequences, n, mode='first'):
        truncated_sequences = []
        for seq in sequences:
            if mode == 'first':
                truncated_sequences.append(seq[:n])
            elif mode == 'last':
                truncated_sequences.append(seq[-n:])
            else:
                raise ValueError("Mode not supported. Use 'first' or 'last'.")
        return truncated_sequences

    def create_features_labels(self, task, world_seq, code_seq, error_seq, case_seq, success_seq=None):
        X, y = [], []
        task_mapping_corr = {}
        task_mapping_success = {}
        
        if self.one_hot:
            world_seq, _ = one_hot_encode_sequences(world_seq)
            code_seq, _ = one_hot_encode_sequences(code_seq)
            error_seq, _ = one_hot_encode_sequences(error_seq)
            case_seq, _ = one_hot_encode_sequences(case_seq)
            success_seq, _ = one_hot_encode_sequences(success_seq)

            task_mapping_corr = {
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
                "code_case_n_world_n_prev_until": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), ca[:i+1].flatten(), w[:i].flatten()]), w[i]),
                "code_case_n_world_n_prev_only": lambda w, co, e, ca, i: (np.hstack([co[:i+1].flatten(), ca[:i+1].flatten(), w[i-1].flatten()]), w[i]),
            }

            task_mapping_success = {
                "world_code_error": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), e.flatten()]), s[0]),
                "world_code_case": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten(), ca.flatten()]), s[0]), 
                "world_code": lambda w, co, e, ca, s: (np.hstack([w.flatten(), co.flatten()]), s[0]), 
                "world": lambda w, co, e, ca, s: (w.flatten(), s[0]),
                "code": lambda w, co, e, ca, s: (co.flatten(), s[0]),
                "error": lambda w, co, e, ca, s: (e.flatten(), s[0]),
                "case": lambda w, co, e, ca, s: (ca.flatten(), s[0]),
                "code_error": lambda w, co, e, ca, s: (np.hstack([co.flatten(), e.flatten()]), s[0]),
                "world_error": lambda w, co, e, ca, s: (np.hstack([w.flatten(), e.flatten()]), s[0]),
                "code_case": lambda w, co, e, ca, s: (np.hstack([co.flatten(), ca.flatten()]), s[0]),
                "world_case": lambda w, co, e, ca, s: (np.hstack([w.flatten(), ca.flatten()]), s[0])
            }
            
        else: 
            task_mapping_corr = {
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
                "code_case_n_world_n_prev_until": lambda w, c, e, cas, i: (c[:i+1] + cas[:i+1] + w[:i], w[i]),
                "code_case_n_world_n_prev_only": lambda w, c, e, cas, i: (c[:i+1] + cas[:i+1] + [w[i-1]], w[i]),
                }
            
            task_mapping_success = {
                "world_code_error": lambda w, co, e, cas, s: (w + co + e, s[0]),
                "world_code_case": lambda w, co, e, ca, s: (w + co + ca, s[0]), 
                "world_code": lambda w, co, e, ca, s: (w + co, s[0]), 
                "world": lambda w, co, e, ca, s: (w, s[0]),
                "code": lambda w, co, e, ca, s: (co, s[0]),
                "error": lambda w, co, e, ca, s: (e, s[0]),
                "case": lambda w, co, e, ca, s: (ca, s[0]),
                "code_error": lambda w, co, e, ca, s: (co + e, s[0]),
                "world_error": lambda w, co, e, ca, s: (w + e, s[0]), 
                "code_case": lambda w, co, e, ca, s: (co + ca, s[0]),
                "world_case": lambda w, co, e, ca, s: (w + ca, s[0]), 
                "error_world_code": lambda w, co, e, ca, s: (e + w + co, s[0]), 
                "world_error_code": lambda w, co, e, ca, s: (w + e + co, s[0]), 
                "world_code_error_case": lambda w, co, e, ca, s: (w + co + e + ca, s[0]),
                "code_world_error": lambda w, co, e, ca, s: (co + w + e, s[0]), # Could add other cases
            }
            
            
        if self.prediction == 'success':
            for world, code, error, case, success in zip(world_seq, code_seq, error_seq, case_seq, success_seq):
                features, label = task_mapping_success[task](world, code, error, case, success)
                X.append(features)
                y.append(label)
                    
        else: 
            for world, code, error, case in zip(world_seq, code_seq, error_seq, case_seq):
                for i in range(1, len(world)):
                    features, label = task_mapping_corr[task](world, code, error, case, i)
                    X.append(features)
                    y.append(label)

        return X, y
