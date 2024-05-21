import random

def stub_evaluation_function(feature_set):
    # This function returns a random accuracy for a given feature set
    return random.uniform(0, 100)

def forward_selection(total_features):
    current_features = []
    best_accuracy = 0
    best_features = []

    print(f"\nUsing no features and “random” evaluation, I get an accuracy of {stub_evaluation_function(current_features):.1f}%")
    print("Beginning search.")

    for i in range(total_features):
        feature_to_add = None
        best_local_accuracy = 0
        
        for feature in range(1, total_features + 1):
            if feature not in current_features:
                new_features = current_features + [feature]
                accuracy = stub_evaluation_function(new_features)
                print(f"Using feature(s) {new_features} accuracy is {accuracy:.1f}%")
                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    feature_to_add = feature
        
        if feature_to_add:
            current_features.append(feature_to_add)
            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_features = current_features.copy()
            print(f"\nFeature set {current_features} was best, accuracy is {best_local_accuracy:.1f}%")
        else:
            break

    print(f"\nFinished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.1f}%\n")

def backward_elimination(total_features):
    current_features = list(range(1, total_features + 1))
    best_accuracy = stub_evaluation_function(current_features)
    best_features = current_features.copy()

    print(f"\nUsing all features and “random” evaluation, I get an accuracy of {best_accuracy:.1f}%")
    print("Beginning search.")

    while len(current_features) > 1:
        feature_to_remove = None
        best_local_accuracy = 0

        for feature in current_features:
            new_features = [f for f in current_features if f != feature]
            accuracy = stub_evaluation_function(new_features)
            print(f"Using feature(s) {new_features} accuracy is {accuracy:.1f}%")
            if accuracy > best_local_accuracy:
                best_local_accuracy = accuracy
                feature_to_remove = feature
        
        if feature_to_remove:
            current_features.remove(feature_to_remove)
            if best_local_accuracy > best_accuracy:
                best_accuracy = best_local_accuracy
                best_features = current_features.copy()
                print(f"\nFeature set {current_features} was best, accuracy is {best_accuracy:.1f}%")
        else:
            break

    print(f"\nFinished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.1f}%\n")

if __name__ == "__main__":
    print("Welcome to Bertie Woosters (change this to your name) Feature Selection Algorithm.")
    total_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("\n1. Forward Selection\n2. Backward Elimination\n3. Bertie's Special Algorithm.")
    choice = int(input())

    if choice == 1:
        forward_selection(total_features)
    elif choice == 2:
        backward_elimination(total_features)
    else:
        print("Bertie's Special Algorithm is not implemented yet.")
