import numpy as np
import time

class NNClassifier:
    def __init__(self, k=1):
        self.trainingData = None
        self.trainingLabels = None
        self.minVal = None
        self.maxVal = None
        self.k = k  # Number of neighbors to consider

    def loadData(self, filepath): # loads features and labels 
        data = np.loadtxt(filepath)
        labels = data[:, 0]
        features = data[:, 1:]
        return labels, features

    def normalize(self, data): # normalize data based off of formula normalized data = (data - Max Value) / (max Value - min Value)
        self.minVal = data.min(axis=0)
        self.maxVal = data.max(axis=0)
        return (data - self.minVal) / (self.maxVal - self.minVal)

    def calcEuclideanDist(self, p1, p2): # Calculate euclidean distance sqrt(summation(qi - pi)^2), since it is 2D we only need to worry about 2 points, p1 and p2
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def train(self, trainInstances, labels): #
        self.trainingData = np.array(trainInstances)
        self.trainingLabels = np.array(labels)
        self.trainingData = self.normalize(self.trainingData)

    def test(self, testInstance):
        if self.trainingData is None or self.trainingData.size == 0: # error catch 
            raise ValueError("The classifier has not been trained yet.")
        
        testInstance = np.array(testInstance) 
        if testInstance.shape[0] != self.trainingData.shape[1]: # error catch 
            raise ValueError(f"Test instance must have {self.trainingData.shape[1]} features, but got {testInstance.shape[0]} features.")
        
        normalizedTestInstance = classifier.normalize(testInstance)
        distances = []

        for idx, trainingInstance in enumerate(self.trainingData): # loop through each of the training data, calculate the euclidean distanve between the two points and return values to an array
            distance = self.calcEuclideanDist(trainingInstance, normalizedTestInstance)
            distances.append((distance, self.trainingLabels[idx]))
        
        distances.sort(key=lambda x: x[0]) # 
        nearestLabels = [label for _, label in distances[:self.k]]
        predictedLabel = max(set(nearestLabels), key=nearestLabels.count)

        return predictedLabel
    

    def leaveOneOutCrossValidation(self, data, labels, featureIndices):
        correctPredictions = 0
        totalInstances = data.shape[0]
        traceInfo = []

        for i in range(totalInstances):
            startTime = time.time()
            
            testInstance = data[i, featureIndices]
            testLabel = labels[i]
            trainData = np.delete(data, i, axis=0)[:, featureIndices]
            trainLabels = np.delete(labels, i)

            self.train(trainData, trainLabels)
            predictedLabel = self.test(testInstance)

            endTime = time.time()
            duration = endTime - startTime

            traceInfo.append({
                "testInstanceIndex": i,
                "testInstance": testInstance,
                "testLabel": testLabel,
                "predicted_label": predictedLabel,
                "correctPrediction": predictedLabel == testLabel,
                "durationSeconds": duration
            })

            if predictedLabel == testLabel:
                correctPredictions += 1

        accuracy = correctPredictions / totalInstances
        return accuracy, traceInfo


filePath = 'small-test-dataset.txt'
classifier = NNClassifier()
labels, features = classifier.loadData(filePath)
featureIndices = [3, 5, 7]  

accuracy, trace_info = classifier.leaveOneOutCrossValidation(features, labels, featureIndices)

print(f"Accuracy: {accuracy * 100}%")
#print("\nTrace Info:")
for trace in trace_info:
    print(trace)
