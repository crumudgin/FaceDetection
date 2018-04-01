from cnn import CNN
import numpy as np
import cv2


face = cv2.imread("trainer/zac/zac0.jpg", 1)
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
network = CNN(.001, face.shape, 10)
trainData = np.array([face.flatten()])
testData = np.array([face.flatten()])
trainLabels = np.array([[0,0,0,0,0,0,0,0,0,1]])
testLabels = np.array([[0,0,0,0,0,0,0,0,0,1]])
network.setNetwork(1, 2, 1, 90*90)
network.train(10, trainData, trainLabels, testData, testLabels)
network.run(testData)