import numpy as np
import torch
import time
import os
import zipfile
import math
device = torch.device('cpu')

# 3000 Decay:
# 7000, 4e-7 = 0.707 - WINNER

# Set Hyperparameters
num_epochs = 7000
learning_rate = 4e-7
decay_factor = 3000

# Set Dimensions of Each Layer
input_dimension = 28*28
hidden_dimension_1 = 200
hidden_dimension_2 = 50
output_dimension = 10

# Define Fraction of Training Data to be Used (1 is all training data, 0 is no training data, etc.)
training_data_fraction = 1

# Entire Training Dataset Consists of 5000 Samples
TRAINING_DATASET_SIZE = 5000
num_training_samples = TRAINING_DATASET_SIZE*training_data_fraction
if num_training_samples > TRAINING_DATASET_SIZE or num_training_samples < 0:
    num_training_samples = TRAINING_DATASET_SIZE

# Entire Test Dataset Consists of 1000 Samples
num_test_samples = 1000

# Define Dimensions of Images
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


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
rawTrainingData = loadDataFile("digitdata/trainingimages", num_training_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
trainingLabels = loadLabelsFile("digitdata/traininglabels", num_training_samples)
rawValidationData = loadDataFile("digitdata/validationimages", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
validationLabels = loadLabelsFile("digitdata/validationlabels", num_test_samples)
rawTestData = loadDataFile("digitdata/testimages", num_test_samples, IMAGE_WIDTH, IMAGE_HEIGHT)
testLabels = loadLabelsFile("digitdata/testlabels", num_test_samples)


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