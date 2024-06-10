import numpy as np
import random

class DataHandler:
    def load_data(self, filename):
        data = np.loadtxt(filename)
        attributes = data[:, 1:]
        targets = data[:, 0]
        return attributes, targets

class KNNClassifier:
    def predict(self, train_data, train_targets, instance):
        distances = np.linalg.norm(train_data - instance, axis=1)
        nearest_index = np.argmin(distances)
        return train_targets[nearest_index]

class Evaluation:
    def __init__(self, knn):
        self.knn = knn

    def leave_one_out_single(self, i, attributes, targets, selected_attributes):
        train_attributes = np.delete(attributes, i, axis=0)[:, selected_attributes]
        train_targets = np.delete(targets, i)
        test_instance = attributes[i, selected_attributes]
        prediction = self.knn.predict(train_attributes, train_targets, test_instance)
        return prediction == targets[i]

    def leave_one_out(self, attributes, targets, selected_attributes):
        selected_attributes = np.array(selected_attributes)
        n_samples = attributes.shape[0]
        
        correct_predictions = sum(
            self.leave_one_out_single(i, attributes, targets, selected_attributes)
            for i in range(n_samples)
        )
        
        accuracy = correct_predictions / n_samples
        return accuracy

    def stub_evaluation(self, attributes, targets, selected_attributes):
        return random.uniform(0, 1)

def greedy_forward_selection(data_handler, knn, evaluation, filename, use_stub=False):
    attributes, targets = data_handler.load_data(filename)
    n_attributes = attributes.shape[1]
    selected_attributes = []
    highest_accuracy = 0
    
    eval_function = evaluation.stub_evaluation if use_stub else evaluation.leave_one_out

    for _ in range(n_attributes):
        best_attribute = None
        local_best_accuracy = 0
        for attribute in range(n_attributes):
            if attribute not in selected_attributes:
                accuracy = eval_function(attributes, targets, selected_attributes + [attribute])
                if accuracy > local_best_accuracy:
                    local_best_accuracy = accuracy
                    best_attribute = attribute
        if best_attribute is not None:
            selected_attributes.append(best_attribute)
            highest_accuracy = local_best_accuracy
    
    return selected_attributes, highest_accuracy

def greedy_backward_elimination(data_handler, knn, evaluation, filename, use_stub=False):
    attributes, targets = data_handler.load_data(filename)
    n_attributes = attributes.shape[1]
    selected_attributes = list(range(n_attributes))
    highest_accuracy = evaluation.leave_one_out(attributes, targets, selected_attributes)
    
    eval_function = evaluation.stub_evaluation if use_stub else evaluation.leave_one_out

    while len(selected_attributes) > 1:
        worst_attribute = None
        local_best_accuracy = 0
        for attribute in selected_attributes:
            temp_attributes = selected_attributes.copy()
            temp_attributes.remove(attribute)
            accuracy = eval_function(attributes, targets, temp_attributes)
            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                worst_attribute = attribute
        if worst_attribute is not None:
            selected_attributes.remove(worst_attribute)
            highest_accuracy = local_best_accuracy
    
    return selected_attributes, highest_accuracy
