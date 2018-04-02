from cnn import CNN
from random import randint
import numpy as np
import cv2
import os

def genFaces(path, name, nameVal, data):
	res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	for filename in os.listdir(path):
		res[nameVal] = 1
		data["trainer/%s/%s" %(name, filename)] = res
		# print(name, data["trainer/%s/%s" %(name, filename)])
	return data

def genAllData():
	data = {}
	for index, filename in enumerate(os.listdir("trainer")):
		# print(index)
		data = genFaces("trainer/%s" %filename, filename, index, data)
	return data

def scale(image):
	return cv2.resize(image, (334, 334))

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
face = cv2.imread("trainer/zac/zac2.jpg", 1)
print(face.shape)
network = CNN(.001, face.shape, 10)
network.setNetwork(1, 2, 2, 84*84)
network.train(1, trainingData, trainingLabels, testingData, testingLabels)
# network.run([trainingData[0]])
# print(trainingLabels[0])


# side = cv2.CascadeClassifier("sideFace.xml")
# front = cv2.CascadeClassifier("FrontalFace.xml")
# cam = cv2.VideoCapture(1)
# while(True):
#     ret, img = cam.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     rightside = side.detectMultiScale(gray, 1.2,5)
#     horizontal_img = cv2.flip(gray, 1)
#     leftside = side.detectMultiScale(horizontal_img, 1.2, 5)
#     frontFace = front.detectMultiScale(gray, 1.2, 5)
#     picCount= 0
#     for (x,y, w, h) in frontFace:
#         face = img[y:y + h, x: x + w]
#         picCount += 1
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     for (x, y, w, h) in rightside:
#         face = img[y:y + h, x: x + w]
#         picCount += 1
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     for (x, y, w, h) in leftside:
#         face = img[y:y + h, x: x + w]
#         picCount += 1
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.imshow("faces", img)
#     face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face = scale(face)
#     print(picCount)
#     currImg = np.array([face.flatten()])
#     if picCount >= 1:
#     	network.run(currImg)
#     if(cv2.waitKey(1) == ord('q')):
#         break

# cam.release()
# cv2.destroyAllWindows()