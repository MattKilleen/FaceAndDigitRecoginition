import numpy as np
import torch
import time
import zipfile
import os
from random import seed
from random import randrange

device = torch.device('cpu')

# Define Dimension of Output Vector
output_dimension = 10

# Define Fraction of Training Data to be Used (1 is all training data, 0 is no training data, etc.)
training_data_fraction = 1

# Entire Training Dataset Consists of 5000 Samples
TRAINING_DATASET_SIZE = 5000
num_training_samples = TRAINING_DATASET_SIZE * training_data_fraction
if num_training_samples > TRAINING_DATASET_SIZE or num_training_samples < 0:
    num_training_samples = TRAINING_DATASET_SIZE

# Entire Test Dataset Consists of 1000 Samples
num_test_samples = 1000

# Define Dimensions of Images
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


# Method to Read Data from Zip - borrowed from Berkeley code
def readlines(filename):
    if (os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n'.encode())


# Method to Load Data from File - borrowed from Berkeley code
def loadDataFile(filename, n, width, height):
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(data)
    return items


# Method to Load Labels from File - borrowed from Berkeley code
def loadLabelsFile(filename, n):
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


# Import Data
rawTrainingData = loadDataFile("digitdata/trainingimages", num_training_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
trainingLabels = loadLabelsFile("digitdata/traininglabels", num_training_samples)
rawValidationData = loadDataFile("digitdata/validationimages", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
validationLabels = loadLabelsFile("digitdata/validationlabels", num_test_samples)
rawTestData = loadDataFile("digitdata/testimages", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
testLabels = loadLabelsFile("digitdata/testlabels", num_test_samples)

# Preprocess dataset into numpy
y_train = np.zeros((len(trainingLabels), output_dimension))
y_test = np.zeros((len(testLabels), output_dimension))

for i in range(len(trainingLabels)):
    x = trainingLabels[i]
    y_train[i][x] = 1

for i in range(len(testLabels)):
    x = testLabels[i]
    y_test[i][x] = 1

x_train_raw = np.asarray(rawTrainingData)
y_train = np.asarray(y_train)
x_test_raw = np.asarray(rawTestData)
y_test = np.asarray(y_test)

x_train = np.zeros((x_train_raw.shape[0], x_train_raw.shape[1] * x_train_raw.shape[2]))
x_test = np.zeros((x_test_raw.shape[0], x_test_raw.shape[1] * x_test_raw.shape[2]))

for i in range(len(x_train_raw)):
    x_train[i] = x_train_raw[i].flatten()

for i in range(len(x_test_raw)):
    x_test[i] = x_test_raw[i].flatten()

### YOUR CODE TO SET UP MODEL STRUCTURE HERE

num_classes = 10

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
    while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
    fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
    return scores


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
    error = row[-1] - prediction
    weights[0] = weights[0] + l_rate * error
    for i in range(len(row) - 1):
        weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
    predictions.append(prediction)
    return (predictions)

# Record Training Start Time
start_time = time.time()

### YOUR TRAIN AND TEST CODE HERE
# Test the Perceptron algorithm on the sonar dataset
# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(x_test, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
