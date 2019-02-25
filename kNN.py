# Example of kNN implememnted from scratch in Python(python 2.7)

import csv
import random
import math
import operator


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1, len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    # We Get the Euclidean Distance directly
    # is the square root of the sum of the squared differences between the two arrays of numbers
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    #  the getNeighbors function that returns k most similar neighbors from the training set for a given test instance
    #  (using the already defined euclideanDistance function)
    #  locate the most similar neighbors for a test instance
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    #  a function for getting the majority voted response from a number of neighbors.
    #  It assumes the class is the last attribute for each neighbor.
    #  devise a predicted response based on those neighbors
    #  We can do this by allowing each neighbor to vote for
    #  their class attribute, and take the majority vote as the prediction
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def getAccuracy(testSet, predictions):
    #   function that sums the total correct predictions and returns the accuracy as a percentage of correct
    # classifications
    #  the accuracy of the model is to calculate a ratio of the total correct predictions out of all predictions made,
    #  called the classification accuracy.
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
