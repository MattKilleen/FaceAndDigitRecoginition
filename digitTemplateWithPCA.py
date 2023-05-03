import numpy as np
import torch
import time
import zipfile
import os
import matplotlib.pyplot as plt
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

x_train_unextracted = np.zeros((x_train_raw.shape[0], x_train_raw.shape[1]*x_train_raw.shape[2]))
x_test_unextracted = np.zeros((x_test_raw.shape[0], x_test_raw.shape[1]*x_test_raw.shape[2]))

for i in range(len(x_train_raw)):
    x_train_unextracted[i] = x_train_raw[i].flatten()

for i in range(len(x_test_raw)):
    x_test_unextracted[i] = x_test_raw[i].flatten()


### The following code computes the mean value for each variable, then subtracts it from each sample for that variable

# Set up the total value and mean vectors, and the centered data matrix
total = np.zeros(len(x_train_unextracted[0]))
mean = np.zeros(len(x_train_unextracted[0]))
x_train_centered = np.zeros((len(x_train_unextracted), len(x_train_unextracted[0])))
x_test_centered = np.zeros((len(x_test_unextracted), len(x_test_unextracted[0])))

# Iterate through each of the values in each of the columns and sum their values to compute the total sum for each column,
for i in range(len(x_train_unextracted)):
    for j in range(len(x_train_unextracted[i])):
        total[j] = total[j] + x_train_unextracted[i][j]

# Iterate through each of the columns, computing the mean by dividing the total for that column by the number of values
# Then subtract the mean from each value in that column
for m in range(len(x_train_unextracted[0])):
    mean[m] = total[m] / (len(x_train_unextracted) * len(x_train_unextracted[0]))
    for n in range(len(x_train_unextracted)):
        x_train_centered[n][m] = x_train_unextracted[n][m] - mean[m]

for m in range(len(x_test_unextracted[0])):
    for n in range(len(x_test_unextracted)):
        x_test_centered[n][m] = x_test_unextracted[n][m] - mean[m]

# Compute the singular value decomposition of the centered data matrix
svd_x_train_centered = np.linalg.svd(x_train_centered)

x = []
num_singular_values = []

# compute the total energy in the dataset
total = 0
for i in range(len(svd_x_train_centered[1])):
    total += svd_x_train_centered[1][i] ** 2
    x.append(svd_x_train_centered[1][i])
    num_singular_values.append(i+1)

# Plot scree plot with absolute scale
plt.title('Scree Plot for the Dataset with Absolute Scale')
plt.xlabel('$i$')
plt.ylabel('Value of the $i$th singular value, $\sigma_i$')
plt.bar(num_singular_values[0:10], x[0:10])
plt.show()

# Plot scree plot with logarithmic scale
fig, ax = plt.subplots()
ax.set_yscale('log')
plt.title('Scree Plot for the Dataset with Logarithmic Scale')
plt.xlabel('$i$')
plt.ylabel('Value of the $i$th singular value, $\sigma_i$')
plt.bar(num_singular_values[0:10], x[0:10])
plt.show()

# compute k such that the top-k principal components of the dataset capture at least 99% of the total energy
sum = 0
k = 0
while k < 2:
    sum += svd_x_train_centered[1][k] ** 2
    k = k + 1

print("2 Principal Components Capture " + str(round(100*sum/total, 2)) + "% of the Total Energy")

# Generate the PCA subspace using the singular value decomposition computed earlier
U = np.zeros((len(svd_x_train_centered[2]), k))

for i in range(len(svd_x_train_centered[2])):
    for j in range(k):
        U[i][j] = svd_x_train_centered[2][i][j]

x_train = np.matmul(x_train_centered, U)
x_test = np.matmul(x_test_centered, U)

### YOUR CODE TO SET UP MODEL STRUCTURE HERE

# Record Training Start Time
start_time = time.time()

### YOUR TRAIN AND TEST CODE HERE