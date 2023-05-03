import numpy as np
import time
import zipfile
import os

#Gaussian function
def gaussian(x, mean, std):
    return (1.0 / (np.sqrt(2*np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std)**2)

def PDF(input):
   #estimate params
   mean = np.mean(input)
   std = np.std(input)
   print(mean, std)
   # fit distribution
   dist = np.norm(mean, std)
   return dist

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

x_train = np.zeros((x_train_raw.shape[0], x_train_raw.shape[1]*x_train_raw.shape[2]))
x_test = np.zeros((x_test_raw.shape[0], x_test_raw.shape[1]*x_test_raw.shape[2]))

for i in range(len(x_train_raw)):
    x_train[i] = x_train_raw[i].flatten()

for i in range(len(x_test_raw)):
    x_test[i] = x_test_raw[i].flatten()

### SETTING UP MODEL STRUCTURE - NAIVE BAYES

#x_train shape is 5000 x 784 (28*28 = 784)
#y_train shape is 5000 x 10 (one-hot-encoded labels)

#The number of classes for the digits, 0-9
num_classes = 10

#List of zeros of length num_classes, to be used in calculating priors
priors = np.zeros(num_classes)

#List of probability distribution functions to be calculated
probabilities = np.zeros((num_classes, x_train.shape[1]))

print(probabilities.shape)

### TRAINING

# Record Training Start Time
start_time = time.time()

### CALCULATING PRIORS

#Iterating through the one-hot-encoded labels of y_train
for labelEncode in y_train:
   
   #Getting the index of the label in the one-hot-encoded y_train labels
   idx = np.where(labelEncode == 1)[0] 

   #Adding 1 to the sum of the respective label in the priors array
   priors[idx] += 1 

#Iterating through each sum to calculate the priors
for x in range(num_classes):
   
   #Calculating the priors by taking the sum of each label and dividing by total number of samples
   priors[x] = priors[x] / TRAINING_DATASET_SIZE

### CALCULATING PROBABILITY DISTRIBUTION FUNCTIONS


# Record Training End Time
end_time = time.time()

# Print Time Taken
print("Time taken: {:.2f} seconds".format(end_time - start_time))