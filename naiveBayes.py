# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    
    util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter() #Log-joint counter, stores each log-probability
    
    "*** YOUR CODE HERE ***"

    #Iterate through each legal label and return the log-join distribution
    for legalLabel in self.legalLabels:

      #Initial value for the joint value to be added to logJoint
      joint = math.log(self.prior[legalLabel])

      #Iterating through datum for features + values
      for feature, value in datum.items():

        prob = self.conditionalProb[feature,legalLabel] #P(feature | legalLabel)

        if value <= 0: #If the value is less than or equal to 0, add the log of !prob
          joint += math.log(1 - prob)

        else: #If the value is greater than 0, add the log of the prob
          joint += math.log(prob)
      
      #Set the logJoint value to the final joint value calculated above
      logJoint[legalLabel] = joint

    #util.raiseNotDefined() # <-- Not sure if needed still!
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = [] #List of features' odds ratios
       
    "*** YOUR CODE HERE ***"

    for feature in self.features:

      cond1 = self.conditionalProb[feature, label1] #P(feature = 1 | label1)
      cond2 = self.conditionalProb[feature, label2] #P(feature = 1 | label2)
      oddsRatio = cond1 / cond2 #Computing the odds ratio

      featuresOdds.append((oddsRatio, feature)) #Appending odds ratio to featuresOdds list

    featuresOdds.sort() #Using Python .sort() function to sort the features in ascending order

    #featuresOdds.reverse() #Flips featuresOdds to descending order

    bestHundred = featuresOdds[:100] #SHOULD fetch 100 best features
    # ^ IF THIS DOESN'T WORK THEN UNCOMMENT THE REVERSE FUNCTION AND TRY THAT!

    #I'm not sure if this will 100% work--the featuresOdds list is 2D I think and I'm not
    #sure if we can access the first 100 like that. I'm also not sure if they only want
    #the best 100 features without the oddsRatio

    #If it needs to only be the features... maybe we can use:
    #for oddsRatio, feature in featuresOdds[:100]:
    #  bestHundred.append(feature)

    #util.raiseNotDefined() # <-- Not sure if needed still!

    return featuresOdds