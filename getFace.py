from random import randint
import numpy as np
import cv2
import os
import random

def genFaces(path, name, nameVal, data):
    for filename in os.listdir(path):
        data.append("alligned/%s/%s" %(name, filename))
    return data

def genAllData():
    data = {}
    for filename in os.listdir("alligned"):
        data[filename] = []
    for index, filename in enumerate(os.listdir("alligned")):
        data[filename] = genFaces("alligned/%s" %filename, filename, index, data[filename])
    return data

def scale(image):
    return cv2.resize(image, (224, 224))

def removeTrainingData(data, labels, percent):
    counter = 1
    truePercent = len(data)//percent
    training = []
    test = []
    trainingLabels = []
    testingLabels = []
    for i in range(len(data)):
        if counter % truePercent == 0:
            test.append(data[i])
            testingLabels.append(labels[i])
        else:
            training.append(data[i])
            trainingLabels.append(labels[i])
        counter += 1
    return training, test, trainingLabels, testingLabels

def cvtData(data):
    res = []
    labels = []
    for name in data:
        for i in data[name]:
            res.append(scale(cv2.imread(i)).flatten())
            labels.append(name)
    return res, labels


def formTripplets(data, labels):
    tripData = []
    tripLabel = []
    counter = 0
    for a in range(len(data)):
        for p in range(len(data)):
            if a != p:
                for n in range(len(data)):
                    if labels[a] != labels[n]:
                        tripData.append((data[a], data[p], data[n]))
                        tripLabel.append((labels[a], labels[n]))
    return tripData, tripLabel

def shuffle(data, labels):
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)
    return data, labels