import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class DataHandler:
    def load_data(self, filename):
        data = pd.read_csv(filename, delim_whitespace=True, header=None)
        features = data.iloc[:, 1:]
        labels = data.iloc[:, 0]
        return features, labels

class NNClassifier:
    def train(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, test_instance):
        distances = np.linalg.norm(self.training_data.values - test_instance.values, axis=1)
        nearest_index = np.argmin(distances)
        return self.training_labels.iloc[nearest_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_single(self, i, data, labels, feature_subset):
        training_data = data.drop(index=i)
        training_labels = labels.drop(index=i)

        self.classifier.train(training_data.iloc[:, feature_subset], training_labels)
        prediction = self.classifier.predict(data.iloc[i, feature_subset])
        
        return prediction == labels.iloc[i]

    def leave_one_out_validation(self, data, labels, feature_subset):
        correct_predictions = 0
        total_instances = len(data)

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.leave_one_out_single, range(total_instances), [data]*total_instances, [labels]*total_instances, [feature_subset]*total_instances)
        
        correct_predictions = sum(results)
        accuracy = correct_predictions / total_instances
        return accuracy

def forward_selection(data_handler, classifier, validator, filename, max_features=None):
    features, labels = data_handler.load_data(filename)
    n_features = features.shape[1]
    selected_features = []
    best_accuracy = 0

    print("\nBeginning search.\n")

    while max_features is None or len(selected_features) < max_features:
        best_local_accuracy = 0
        best_feature = None

        for feature in range(n_features):
            if feature not in selected_features:
                accuracy = validator.leave_one_out_validation(features, labels, selected_features + [feature])
                print(f"Using feature(s) {selected_features + [feature]} accuracy is {accuracy:.4f}")
                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            best_accuracy = best_local_accuracy
            print(f"\nFeature set {selected_features} was best, accuracy is {best_accuracy:.4f}")

    print(f"\nFinished search!! The best feature subset is {selected_features}, which has an accuracy of {best_accuracy:.4f}\n")

def backward_elimination(data_handler, classifier, validator, filename, min_features=1):
    features, labels = data_handler.load_data(filename)
    n_features = features.shape[1]
    selected_features = list(range(n_features))
    best_accuracy = validator.leave_one_out_validation(features, labels, selected_features)

    print("\nBeginning search.\n")

    while len(selected_features) > min_features:
        worst_feature = None
        best_local_accuracy = 0

        for feature in selected_features:
            temp_features = selected_features.copy()
            temp_features.remove(feature)
            accuracy = validator.leave_one_out_validation(features, labels, temp_features)
            print(f"Using feature(s) {temp_features} accuracy is {accuracy:.4f}")
            if accuracy > best_local_accuracy:
                best_local_accuracy = accuracy
                worst_feature = feature

        if worst_feature is not None:
            selected_features.remove(worst_feature)
            best_accuracy = best_local_accuracy
            print(f"\nFeature set {selected_features} was best, accuracy is {best_accuracy:.4f}")

    print(f"\nFinished search!! The best feature subset is {selected_features}, which has an accuracy of {best_accuracy:.4f}\n")

