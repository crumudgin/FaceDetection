from cnn import CNN
import network
from random import randint
import numpy as np
import cv2
import os
import tensorflow as tf
from getFace import *

data= genAllData()
data, labels = cvtData(data)
trainingData, testingData, trainingLabels, testingLabels = removeTrainingData(data, labels, 10)
trainingData, trainingLabels = shuffle(trainingData, trainingLabels)
trainingData, trainingLabels = formTripplets(trainingData, trainingLabels)
testingData, testingLabels = formTripplets(testingData, testingLabels)


network = CNN(.001, (224, 224), 128)
network.setNetwork(1, 3, 2, 84*84, 3136 * 16 * 4)
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    initOptimiser = tf.global_variables_initializer()
    sess.run(initOptimiser)
    network.trippletTrain(1, trainingData, trainingLabels, testingData, sess)
    prediction = network.finalOut
    network.loadFaces(sess, 
        [testingData[0][0], testingData[145][0], testingData[290][0], testingData[435][0], testingData[590][0]], 
        [testingLabels[0][0], testingLabels[145][0], testingLabels[290][0], testingLabels[435][0], testingLabels[590][0]])
    print(len(testingLabels))
    print(testingLabels[0])
    print(testingLabels[145])
    print(testingLabels[290])
    print(testingLabels[435])
    print(testingLabels[590])
    # anchor = network.trippletRun(sess, [testingData[0][0]], prediction)
    network.trippletRun(sess, [testingData[0][1]], prediction)
    network.trippletRun(sess, [testingData[145][1]], prediction)
    network.trippletRun(sess, [testingData[290][1]], prediction)
    network.trippletRun(sess, [testingData[435][1]], prediction)
    network.trippletRun(sess, [testingData[590][1]], prediction)