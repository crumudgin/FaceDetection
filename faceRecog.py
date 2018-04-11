from cnn import CNN
import network
from random import randint
import numpy as np
import cv2
import os
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def genFaces(path, name, nameVal, data):
    for filename in os.listdir(path):
        data["alligned/%s/%s" %(name, filename)] = name
        # print(name, data["alligned/%s/%s" %(name, filename)])
    return data

def genAllData():
    data = {}
    for index, filename in enumerate(os.listdir("alligned")):
        # print(index)
        data = genFaces("alligned/%s" %filename, filename, index, data)
    return data

def scale(image):
    return cv2.resize(image, (224, 224))

def removeTrainingData(data, percent):
    counter = 1
    truePercent = len(data)//percent
    test = {}
    training = {}
    for i in data:
        if counter % truePercent == 0:
            test[i] = data[i]
        else:
            training[i] = data[i]
        counter += 1
    return training, test

def cvtData(data):
    labels = []
    res = []
    for i in data:
        img = cv2.imread(i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = scale(img)
        labels.append(data[i])
        res.append(img.flatten())
    return res, labels

def formTripplets(data, labels):
    tripData = []
    tripLabel = []
    for i, anchor in enumerate(data):
        for j, positive in enumerate(data):
            if labels[i] == labels[j]:
                for k, negative in enumerate(data):
                    if labels[i] != labels[k]:
                        tripData.append((anchor, positive, negative))
                        tripLabel.append((labels[i], labels[k]))
    return(tripData, tripLabel)



data = genAllData()
data, test = removeTrainingData(data, 10)
trainingData, trainingLabels = cvtData(data)
trainingData, trainingLabels = formTripplets(trainingData, trainingLabels)
testingData, testingLabels = cvtData(test)
testingData, testingLabels = formTripplets(testingData, testingLabels)
# print(trainingLabels)

network = CNN(.001, (224, 224), 128)
network.setNetwork(1, 3, 2, 84*84, 3136 * 16 * 4)
network.trippletTrain(10, trainingData, trainingLabels, testingData)
# network.train(1, trainingData, trainingLabels, testingData, testingLabels)

# trainingData = mnist.train.images
# trainingLabels = np.asarray(mnist.train.labels, dtype=np.int32)
# testingData = mnist.test.images
# testingLabels = np.asarray(mnist.test.labels, dtype=np.int32)
# print(trainingLabels)

# network = CNN(.001, (28, 28), 10)
# network.setNetwork(1, 3, 2, 84*84, 1024 * 4)
# network.train(10, trainingData, trainingLabels, testingData, testingLabels)