import numpy as np
import tensorflow as tf  # I only used TensorFlow to import the dataset - nothing to do with the actual deep learning task
import torch
import time
import os
import zipfile
import math
device = torch.device('cpu')

# 1000 Decay - 4200,500,30,2:
# 3000, 3e-7 = 0.607


# 3000 Decay - 4200,700,30,2:
# 3000, 3e-7 = 0.627, 0.593


# 10000 Decay - 4200,500,30,2:
# 3000, 3e-7 = 0.627
# 3000, 5e-7 = 0.633


# 100000 Decay - 4200,500,30,2:
# 3000, 3e-7 = 0.513


# 3000 Decay - 4200,1000,200,50,10,2:
# 3000, 3e-7 = 0.513

# Set Hyperparameters
num_epochs = 3000
learning_rate = 5e-7
decay_factor = 3000

# Set Dimensions of Each Layer
input_dimension = 4200
hidden_dimension_1 = 500
hidden_dimension_2 = 30
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


# Preprocess dataset into torch
y_train = np.zeros((len(trainingLabels), output_dimension))
y_test = np.zeros((len(testLabels), output_dimension))

for i in range(len(trainingLabels)):
    x = trainingLabels[i]
    y_train[i][x] = 1

for i in range(len(testLabels)):
    x = testLabels[i]
    y_test[i][x] = 1

x_train = torch.from_numpy(np.asarray(rawTrainingData))
y_train = torch.from_numpy(np.asarray(y_train))
x_test = torch.from_numpy(np.asarray(rawTestData))
y_test = torch.from_numpy(np.asarray(y_test))

x_train = torch.flatten(x_train, start_dim=1).float()
y_train = torch.flatten(y_train, start_dim=1).float()
x_test = torch.flatten(x_test, start_dim=1).float()
y_test = torch.flatten(y_test, start_dim=1).float()

# Set Up Model
model = torch.nn.Sequential(torch.nn.Linear(input_dimension, hidden_dimension_1),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_dimension_1, hidden_dimension_2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_dimension_2, output_dimension),
                            torch.nn.Softmax(dim=1)).to(device)

# Specify Loss Function
loss_fn = torch.nn.MSELoss(reduction='sum')

# Record Training Start Time
start_time = time.time()

# Train the Network
for t in range(num_epochs):

    # Forward Propagate
    y_pred = model(x_train)

    # Compute and Print Loss
    loss = loss_fn(y_pred, y_train)
    print("Epoch " + str(t) + ": Loss = " + str(loss.item()))

    # Back Propagate
    model.zero_grad()
    loss.backward()

    # Update Weights According to Gradients Computed in Backpropagation
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate*math.exp(-t/decay_factor) * param.grad


# Compute Final Training Accuracy

# Forward Propagate

y_pred = model(x_train)

# Count Number of Correct Predicted Labels

num_correct = 0
for i in range(num_training_samples):
    num_correct += y_train[i][int(torch.argmax(y_pred[i]).item())].item()

# Compute Accuracy as Portion of Correct Predicted Labels and Print

accuracy = num_correct / num_training_samples

print("\nFinal Training Accuracy = " + str(accuracy))


# Record and Print Total Training Time
training_time = time.time() - start_time
print("\nTotal Training time is " + str(training_time) + " seconds")


# Evaluate Performance on Test Data

# Forward Propagate

y_pred = model(x_test)

# Count Number of Correct Predicted Labels

num_correct = 0
for i in range(num_test_samples):
    num_correct += y_test[i][int(torch.argmax(y_pred[i]).item())].item()

# Compute Accuracy as Portion of Correct Predicted Labels and Print

accuracy = num_correct / num_test_samples

print("\nTest Accuracy = " + str(accuracy))