import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

class NNClassifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None

    def train(self, X, y):
        self.training_data = X
        self.training_labels = y

    def test(self, x):
        distances = np.linalg.norm(self.training_data - x, axis=1)
        nearest_index = np.argmin(distances)
        return self.training_labels[nearest_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate(self, X, y):
        correct_predictions = 0
        for i in range(len(X)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            x_test = X[i]
            y_test = y[i]

            self.classifier.train(X_train, y_train)
            y_pred = self.classifier.test(x_test)

            if y_pred == y_test:
                correct_predictions += 1

        accuracy = correct_predictions / len(X)
        return accuracy

# Function to load data from the file
def loadData(filePath, featureIndices):
    data = np.loadtxt(filePath)
    X = data[:, featureIndices]  # Select specified features
    y = data[:, 0]               # Class labels are in the first column
    return X, y

# For 'small-test-dataset.txt' use [3, 5, 7], for 'large-test-dataset.txt' use [1, 15, 27]
featureIndices = [3, 5, 7]

startTime = time.time()

# Load dataset
smallTestData = 'small-test-dataset.txt'
largeTestData = 'large-test-dataset.txt'
loadStart = time.time()
X, y = loadData(smallTestData, featureIndices)
loadEnd = time.time()
print(f"Data loaded in {loadEnd - loadStart:.4f} seconds")

# Normalize the data
normalizeStart = time.time()
scaler = StandardScaler()
X = scaler.fit_transform(X)
normalizeEnd = time.time()
print(f"Normalization of data completed in {normalizeEnd - normalizeStart:.4f} seconds")

# Initialize and evaluate the NN classifier using leave-one-out validation
nn = NNClassifier()
validator = Validator(nn)

evaluateStart = time.time()
accuracy = validator.evaluate(X, y)
evaluateEnd = time.time()
print(f"Evaluate completed in {evaluateEnd - evaluateStart:.4f} seconds")

endTime = time.time()
totalTime = endTime - startTime

print(f'Total time {totalTime:.4f} seconds')
print(f'Accuracy: {accuracy:.2f}')


#our names
