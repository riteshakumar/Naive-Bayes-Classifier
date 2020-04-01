# driver code
import csv
import random
import math

# Calculating Mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Calculating Standard Deviation
def std_dev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# find Mean and Standard Deviation for each instance
def MeanAndStdDev(mydata):
    info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)]
    del info[-1]
    return info

# Group the data rows under each class yes or
# no in dictionary eg: dict[yes] and dict[no]
def groupUnderClass(mydata):
      dict = {}
      for i in range(len(mydata)):
          if (mydata[i][-1] not in dict):
              dict[mydata[i][-1]] = []
          dict[mydata[i][-1]].append(mydata[i])
      return dict

# find Mean and Standard Deviation under each class
def MeanAndStdDevForClass(mydata):
    info = {}
    dict = groupUnderClass(mydata)
    for classValue, instances in dict.items():
        info[classValue] = MeanAndStdDev(instances)
    return info

# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x, mean, stdev):
    try:
        expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo
    # Gaussian approximation when mean and std_dev is 0
    except ZeroDivisionError:
        return 0.01

# Calculate Class Probabilities
def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            #Enhancement : Log Probabilities by adding the log of the probabilities together
            probabilities[classValue] += math.log(calculateGaussianProbability(x, mean, std_dev))
    return probabilities

# Make prediction - highest probability is the prediction
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        probability = math.exp(probability)
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# returns predictions for a set of examples
def getPredictions(info, test):

    predictions = []
    for i in range(len(test)):
        result = predict(info, test[i])
        predictions.append(result)

    return predictions

# Accuracy score
def accuracy_rate(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0

# Splitting the data
def cross_val_splitting(mydata, fold):
    from sklearn.model_selection import KFold
    # prepare cross validation
    kfold = KFold(fold, True, 1)
    # enumerate splits
    for train, test in kfold.split(mydata):
        mydata[train], mydata[test]
    return mydata[train], mydata[test]

# the categorical class names are changed to numeric data
# eg: yes and no encoded to 1 and 0
def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    classes = sorted(classes)
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata

# load the file and store it in mydata
import pandas as pd
mydata=pd.read_csv(r'hayes-roth.csv', header=0)
#mydata=pd.read_csv('car.csv', header=0)
#mydata=pd.read_csv('breast-cancer.csv', header=0)

# pre processing text data to numeric data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in mydata.columns:
    mydata[i] = le.fit_transform(mydata[i])

mydata = mydata.values

mydata = encode_class(mydata)
for i in range(len(mydata)):
    mydata[i] = [float(x) for x in mydata[i]]

# K Fold Cross Validation
fold = 10
train_data, test_data = cross_val_splitting(mydata, fold)
print('Total number of examples are: ', len(mydata))
print('Out of these, training examples are: ', len(train_data))
print("Test examples are: ", len(test_data))

# prepare model
info = MeanAndStdDevForClass(train_data)

# test model
predictions = getPredictions(info, test_data)

#Accuracy
accuracy = accuracy_rate(test_data, predictions)
print("Accuracy of the model is: ", accuracy)