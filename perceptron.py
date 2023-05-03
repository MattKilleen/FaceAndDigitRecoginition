# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util

PRINT = True


class PerceptronClassifier:
    """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights == weights;

    def train(self, trainingData, trainingLabels, validationData=None, validationLabels=None):
        """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

        self.features = list(trainingData[0].keys())

        for i in range(self.max_iterations):
            print("Starting iteration ", i, "...")
            for j in range(len(trainingData)):

                maxLabel = ""
                maxScore = -100
                f = trainingData[j]
                for l in list(set(trainingLabels)):
                    weight = self.weights[l]
                    score = 0
                    for feature, value in f.iteritems():
                        score += value * weight[feature]
                    if score > maxScore:
                        maxScore = score
                        maxLabel = l
                if maxLabel is not trainingLabels[j]:
                    self.weights[trainingLabels[j]] += f
                    self.weights[maxLabel] -= f

    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []
        weights = self.weights[label]
        for i in range(100):
            weight = weights.argMax()
            featuresWeights.append(weight)
            weights[weight] = weights[weight] - 99999999

        return featuresWeights
