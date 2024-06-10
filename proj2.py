import pandas as pd
import numpy as np
import random

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

    def leave_one_out_validation(self, data, labels, feature_subset):
        correct_predictions = 0
        total_instances = len(data)

        for i in range(total_instances):
            training_data = data.drop(index=i)
            training_labels = labels.drop(index=i)

            self.classifier.train(training_data.iloc[:, feature_subset], training_labels)
            prediction = self.classifier.predict(data.iloc[i, feature_subset])
            
            if prediction == labels.iloc[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_instances
        return accuracy

def evaluation_function(feature_subset):
    # Dummy implementation of evaluation function
    # Replace this with the actual logic or function call as needed
    return random.uniform(0, 1) * 100  # Random accuracy for placeholder

def forward_selection(total_features):
    current_features = []
    best_accuracy = 0
    best_features = []

    print(f"\nUsing no features and “random” evaluation, I get an accuracy of {evaluation_function(current_features):.1f}%")
    print("Beginning search.")

    for i in range(total_features):
        feature_to_add = None
        best_local_accuracy = 0
        
        for feature in range(total_features):
            if feature not in current_features:
                new_features = current_features + [feature]
                accuracy = evaluation_function(new_features)
                print(f"Using feature(s) {new_features} accuracy is {accuracy:.4f}")
                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    feature_to_add = feature
        
        if feature_to_add is not None:
            current_features.append(feature_to_add)
            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_features = current_features.copy()
            print(f"\nFeature set {current_features} was best, accuracy is {best_local_accuracy:.4f}")
        else:
            break

    print(f"\nFinished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.4f}\n")

def backward_elimination(total_features):
    current_features = list(range(total_features))
    best_accuracy = evaluation_function(current_features)
    best_features = current_features.copy()

    print(f"\nUsing all features and “random” evaluation, I get an accuracy of {best_accuracy:.4f}")
    print("Beginning search.")

    while len(current_features) > 1:
        feature_to_remove = None
        best_local_accuracy = 0

        for feature in current_features:
            new_features = [f for f in current_features if f != feature]
            accuracy = evaluation_function(new_features)
            print(f"Using feature(s) {new_features} accuracy is {accuracy:.4f}")
            if accuracy > best_local_accuracy:
                best_local_accuracy = accuracy
                feature_to_remove = feature
        
        if feature_to_remove is not None:
            current_features.remove(feature_to_remove)
            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_features = current_features.copy()
                print(f"\nFeature set {current_features} was best, accuracy is {best_accuracy:.4f}")
        else:
            break

    print(f"\nFinished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.4f}\n")

def main():
    data_handler = DataHandler()
    classifier = NNClassifier()
    validator = Validator(classifier)

    # Test on small dataset
    print("\nTesting on CS170_Spring_2024_Small_data__81.txt")
    features, labels = data_handler.load_data("CS170_Spring_2024_Small_data__81.txt")
    accuracy = validator.leave_one_out_validation(features, labels, [0, 1, 2])  # Example features
    print(f"Obtained accuracy: {accuracy:.3f}")

    # Test on large dataset with actual evaluation
    print("\nTesting on CS170_Spring_2024_Large_data__81.txt with features {1, 15, 27}")
    features, labels = data_handler.load_data("CS170_Spring_2024_Large_data__81.txt")
    accuracy = validator.leave_one_out_validation(features, labels, [0, 14, 26])  # Indexing from 0, so 1 -> 0, 15 -> 14, 27 -> 26
    print(f"Expected accuracy: ~0.949, Obtained accuracy: {accuracy:.3f}")

    # Feature selection
    print("Type the number of the algorithm you want to run.")
    print("\n1. Forward Selection\n2. Backward Elimination")
    choice = int(input())

    if choice == 1:
        forward_selection(features.shape[1])
    elif choice == 2:
        backward_elimination(features.shape[1])
    else:
        print("Invalid choice. Please choose either 1 or 2.")

if __name__ == "__main__":
    main()
