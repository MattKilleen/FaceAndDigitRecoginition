import numpy as np
import torch
import time
import zipfile
import os
import perceptron
device = torch.device('cpu')

# Define Dimension of Output Vector
output_dimension = 2

# Define Fraction of Training Data to be Used (1 is all training data, 0 is no training data, etc.)
training_data_fraction = 1

# Entire Training Dataset Consists of 451 Samples
TRAINING_DATASET_SIZE = 451
num_training_samples = TRAINING_DATASET_SIZE*training_data_fraction
if num_training_samples > TRAINING_DATASET_SIZE or num_training_samples < 0:
    num_training_samples = TRAINING_DATASET_SIZE

# Entire Test Dataset Consists of 150 Samples
num_test_samples = 150

# Define Dimensions of Images
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 70


# Method to Read Data from Zip - borrowed from Berkeley code
def readlines(filename):
  if(os.path.exists(filename)):
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
rawTrainingData = loadDataFile("facedata/facedatatrain", num_training_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
trainingLabels = loadLabelsFile("facedata/facedatatrainlabels", num_training_samples)
rawValidationData = loadDataFile("facedata/facedatatrain", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
validationLabels = loadLabelsFile("facedata/facedatatrainlabels", num_test_samples)
rawTestData = loadDataFile("facedata/facedatatest", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
testLabels = loadLabelsFile("facedata/facedatatestlabels", num_test_samples)


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

x_train = np.zeros((x_train_raw.shape[0], x_train_raw.shape[1]*x_train_raw.shape[2]))
x_test = np.zeros((x_test_raw.shape[0], x_test_raw.shape[1]*x_test_raw.shape[2]))

for i in range(len(x_train_raw)):
    x_train[i] = x_train_raw[i].flatten()

for i in range(len(x_test_raw)):
    x_test[i] = x_test_raw[i].flatten()

### YOUR CODE TO SET UP MODEL STRUCTURE HERE

# Record Training Start Time
start_time = time.time()

### YOUR TRAIN AND TEST CODE HERE