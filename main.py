from proj2 import DataHandler, NNClassifier, Validator, forward_selection, backward_elimination

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
        forward_selection(data_handler, classifier, validator, "CS170_Spring_2024_Large_data__81.txt", max_features=5)
    elif choice == 2:
        backward_elimination(data_handler, classifier, validator, "CS170_Spring_2024_Large_data__81.txt", min_features=5)
    else:
        print("Invalid choice. Please choose either 1 or 2.")

if __name__ == "__main__":
    main()
