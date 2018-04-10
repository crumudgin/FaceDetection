from cnn import CNN
import network
from random import randint
import numpy as np
import cv2
import os
import tensorflow as tf 

def genFaces(path, name, nameVal, data):
    res = np.array([0, 0, 0, 0])#, -1, -1, -1, -1, -1, -1]
    for filename in os.listdir(path):
        res[nameVal] = 1
        data["alligned/%s/%s" %(name, filename)] = res
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



data = genAllData()
data, test = removeTrainingData(data, 10)
trainingData, trainingLabels = cvtData(data)
testingData, testingLabels = cvtData(test)
trainingLabels = trainingLabels

# with tf.Graph().as_default():
#   globalStep = tf.train.get_or_create_global_step()
#   data = [tf.cast(i, tf.float16) for i in trainingData]
#   labels = [tf.cast(i, tf.int64) for i in trainingLabels]
#   # print(labels)
#   logits = network.constructNetwork(data, 64*64)
#   # print(logits[0])
#   loss = network.loss(logits, labels)

    # trainOp = network.train(loss, globalStep)
face = cv2.imread("trainer/zac/zac2.jpg", 1)
# print(face.shape)
network = CNN(.001, (224, 224), 4)
network.setNetwork(1, 3, 2, 84*84)
network.train(100, trainingData, trainingLabels, testingData, testingLabels)
# network.run([trainingData[0]])
# print(trainingLabels[0])
# -1.00021213e-02