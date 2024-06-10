from proj2 import DataHandler, KNNClassifier, Evaluation, greedy_forward_selection, greedy_backward_elimination

def main():
    data_handler = DataHandler()
    knn = KNNClassifier()
    evaluation = Evaluation(knn)

    # Test on small dataset with stub evaluation
    print("\nTesting on CS170_Spring_2024_Small_data__81.txt with stub evaluation function:")
    selected_attributes, accuracy = greedy_forward_selection(data_handler, knn, evaluation, 'CS170_Spring_2024_Small_data__81.txt', use_stub=True)
    print(f"Forward Selection - Selected attributes: {selected_attributes}, Stub accuracy: {accuracy:.3f}")

    selected_attributes, accuracy = greedy_backward_elimination(data_handler, knn, evaluation, 'CS170_Spring_2024_Small_data__81.txt', use_stub=True)
    print(f"Backward Elimination - Selected attributes: {selected_attributes}, Stub accuracy: {accuracy:.3f}")

    # Test on small dataset with actual evaluation
    print("\nTesting on CS170_Spring_2024_Small_data__81.txt with actual evaluation function:")
    selected_attributes, accuracy = greedy_forward_selection(data_handler, knn, evaluation, 'CS170_Spring_2024_Small_data__81.txt')
    print(f"Forward Selection - Selected attributes: {selected_attributes}, Actual accuracy: {accuracy:.3f}")

    selected_attributes, accuracy = greedy_backward_elimination(data_handler, knn, evaluation, 'CS170_Spring_2024_Small_data__81.txt')
    print(f"Backward Elimination - Selected attributes: {selected_attributes}, Actual accuracy: {accuracy:.3f}")

    # Test on large dataset with actual evaluation
    print("\nTesting on CS170_Spring_2024_Large_data__81.txt with features {1, 15, 27}")
    attributes, targets = data_handler.load_data('CS170_Spring_2024_Large_data__81.txt')
    accuracy = evaluation.leave_one_out(attributes, targets, [0, 14, 26])  # Indexing from 0, so 1 -> 0, 15 -> 14, 27 -> 26
    print(f"Expected accuracy: ~0.949, Obtained accuracy: {accuracy:.3f}")

    # Large dataset feature selection
    print("\nUsing actual evaluation function on CS170_Spring_2024_Large_data__81.txt:")
    selected_attributes, accuracy = greedy_forward_selection(data_handler, knn, evaluation, 'CS170_Spring_2024_Large_data__81.txt')
    print(f"Forward Selection - Selected attributes: {selected_attributes}, Actual accuracy: {accuracy:.3f}")

    selected_attributes, accuracy = greedy_backward_elimination(data_handler, knn, evaluation, 'CS170_Spring_2024_Large_data__81.txt')
    print(f"Backward Elimination - Selected attributes: {selected_attributes}, Actual accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
