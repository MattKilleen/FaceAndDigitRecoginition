import numpy as np
from scipy.stats import norm
import time
import zipfile
import os
import matplotlib.pyplot as plt

#Function to calculate probability distribution function (Gaussian)
def PDF(input):
   std = np.std(input)
   mean = np.mean(input)
   dist = norm(mean, std)
   return dist

def probability(x, prior, dist1, dist2):
    return prior * dist1.pdf(x[0]) * dist2.pdf(x[1])

def testing(x, prior, dists):
    prob = prior
    for idx, val in enumerate(x):
        prob = prob * dists[idx].pdf(val)
    return prob


# Define Dimension of Output Vector
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
            #print("Truncating at %d examples (maximum)" % i)
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

# compute k such that the top-k principal components of the dataset capture at least 99% of the total energy
sum = 0
k = 0
PCA_DIM = 10
while k < PCA_DIM:
    sum += svd_x_train_centered[1][k] ** 2
    k = k + 1

#print("2 Principal Components Capture " + str(round(100*sum/total, 2)) + "% of the Total Energy")

# Generate the PCA subspace using the singular value decomposition computed earlier
U = np.zeros((len(svd_x_train_centered[2]), k))

for i in range(len(svd_x_train_centered[2])):
    for j in range(k):
        U[i][j] = svd_x_train_centered[2][i][j]

x_train = np.matmul(x_train_centered, U)
x_test = np.matmul(x_test_centered, U)

### SETTING UP MODEL STRUCTURE - NAIVE BAYES

#x_train shape is 5000 x 784 (28*28 = 784)
#y_train shape is 5000 x 10 (one-hot-encoded labels)

#The number of classes for the digits, 0-9
num_classes = 10

#List of zeros of length num_classes, to be used in calculating priors
priors = np.zeros(num_classes)

### TRAINING

# Record Training Start Time
start_time = time.time()

### CALCULATING PRIORS

#Creating empty array of size 10 for each class of identification
classes = np.empty(10, dtype=object)

#Iterating through the one-hot-encoded labels of y_train
for index, labelEncode in enumerate(y_train):
    
    #Getting the index of the label in the one-hot-encoded y_train labels
    idx = int(np.where(labelEncode == 1)[0])

    #Adding 1 to the sum of the respective label in the priors array
    priors[idx] += 1

    #Adding x_train features to respective class in classes array, index-corresponding.
    if classes[idx] is None: #If it's empty (no values added yet)
        classes[idx] = np.array([x_train[index]])
    else:
        classes[idx] = np.append(classes[idx], [x_train[index]], axis=0)

#Iterating through each sum to calculate the priors
for x in range(num_classes):
   
    #Calculating the priors by taking the sum of each label and dividing by total number of samples
    priors[x] = priors[x] / TRAINING_DATASET_SIZE

### CALCULATING PROBABILITY DISTRIBUTION FUNCTIONS

#Empty PDF list of size 10x2
pdfs = [[None for j in range(PCA_DIM)] for i in range(10)]

#Calculating distributions and storing each in the PDF list
for x in range(num_classes):
    for dim in range(PCA_DIM):
        pdfs[x][dim] = PDF(classes[x][:,dim])

#Setting counter for # of correct classifications
correct = 0

#Iterating through the testing data
for index, sample in enumerate(x_test):

    #Creating an empty list of probabilities for each sample
    probabilities = np.zeros(10)

    #Calculating the probabilities based on the distributions and priors that we calculated earlier
    for idx in range(len(probabilities)):
        #probabilities[idx] = probability(sample, priors[idx], pdfs[idx][0], pdfs[idx][1])
        #probabilities[idx] = probabilities[idx]*100 #Multiplied by 100 for accuracy and readability
        probabilities[idx] = testing(sample,priors[idx],pdfs[idx])
        probabilities[idx] = probabilities[idx]*100

    #Fetching index (classification) of the highest probability
    predicted = probabilities.argmax()
    #Fetching the actual classification of the given sample
    actual = int(np.where(y_test[index] == 1)[0])

    #Adding 1 to the correct count if the predicted label matches the actual label
    if(predicted==actual):
        correct += 1

print("Total Correct:", correct)
accuracy = correct / len(y_test)
print(accuracy)

# Record Training End Time
end_time = time.time()

# Print Time Taken
print("Time taken to train: {:.2f} seconds".format(end_time - start_time))
