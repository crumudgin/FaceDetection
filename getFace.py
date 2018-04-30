from random import randint
from math import ceil
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import tables

def genFaces(path, name, nameVal, data):
    for filename in os.listdir(path):
        for image in os.listdir("%s/%s" %(path,filename)):
            # print(image)
            data.append("%s/%s/%s" %(path, filename, image))
    return data

def genAllData(directory):
    data = {}
    count = 0
    for filename in os.listdir(directory):
        data[filename] = []
    for index, filename in enumerate(os.listdir(directory)):
        data[filename] = genFaces("%s/%s" %(directory, filename), filename, index, data[filename])
        count += len(data[filename])
    return data, count

def getClassifierCount(directory):
    count = 0
    for filename in os.listdir(directory):
        count += 1
    return count

def scale(image):
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

def cvtData(data, count):
    res = []
    labels = []
    counter = 0
    for name in data: 
        counter += 1
        print(name, counter)
        for i in data[name]:
            res.append(cv2.imread(i))
            labels.append(name)
    return res, labels

def shuffle(data, labels):
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)
    return data, labels

def getAnchors(data, labels):
    faces = []
    faceLabels = []
    for i in range(len(labels)):
        if labels[i] not in faceLabels:
            faces.append(data[i])
            faceLabels.append(labels[i])
    return faces, faceLabels

def filterFace(directory):
    side = cv2.CascadeClassifier("sideFace.xml")
    front = cv2.CascadeClassifier("FrontalFace.xml")
    picCount = 0
    img = cv2.imread(directory)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rightside = side.detectMultiScale(gray, 1.2,5)
    horizontal_img = cv2.flip(gray, 1)
    leftside = side.detectMultiScale(horizontal_img, 1.2, 5)
    frontFace = front.detectMultiScale(gray, 1.2, 5)
    for (x,y, w, h) in frontFace:
        face = img[y:y + h, x: x + w]
        picCount += 1
        return scale(face)
    for (x, y, w, h) in rightside:
        face = img[y:y + h, x: x + w]
        picCount += 1
        return scale(face)
    for (x, y, w, h) in leftside:
        face = img[y:y + h, x: x + w]
        picCount += 1
        return scale(face)

def getFaces(directory):
    trainingData, count = genAllData(directory)
    saveToHDF5("/media/zac/easystore/dataset.hdf5", trainingData)
    # trainingData, testingLabels = cvtData(trainingData, count)
    # trainingData, trainingLabels = shuffle(trainingData, trainingLabels)
    # testingData, count = genAllData(testingDirectory)
    # testingData, testingLabels = cvtData(trainingData, count)
    # return trainingData, trainingLabels, testingData, testingLabels





def saveToHDF5(filename, data):
    dataLst = []
    labels = []
    for name in data:
        for i in data[name]:
            dataLst.append(i)
            labels.append(name)

    dataLst, labels = shuffle(dataLst, labels)
    trainData = dataLst[0:int(.6*len(dataLst))]
    trainLabels = labels[0:int(.6*len(dataLst))]
    valData = dataLst[int(.6*len(dataLst)):int(.8*len(dataLst))]
    valLabels = labels[int(.6*len(dataLst)):int(.8*len(dataLst))]
    testData = dataLst[int(.8*len(dataLst)):]
    testLabels = labels[int(.8*len(dataLst)):]

    imgDtype = tables.UInt8Atom()
    dataShape = (0, 224, 224, 3)

    file = tables.open_file(filename, mode="w")

    trainStorage = file.create_earray(file.root, "trainImg", imgDtype, shape = dataShape)
    valStorage = file.create_earray(file.root, "valImg", imgDtype, shape = dataShape)
    testStorage = file.create_earray(file.root, "testImg", imgDtype, shape = dataShape)

    # meanStorage = file.create_earray(file.root, "trainMean", imgDtype, shape = dataShape)

    # mean = np.
    labels = []
    counter = 0
    for i in range(len(trainData)):
        if i % 100 == 0:
            print("train data : %d of %d" %(i, len(trainData)))
        face = filterFace(trainData[i])
        if face is not None:
            # print("We got a None for %s" %trainLabels[i])
            counter += 1
            labels.append(trainLabels[i])
            trainStorage.append(face[None])
    file.create_array(file.root, "trainLabels", labels)
    labels = []
    for i in range(len(valData)):
        if i % 1000 == 0:
            print("val data : %d of %d" %(i, len(valData)))
        face = filterFace(valData[i])
        if face is not None:
            # print("We got a None for %s" %valLabels[i])
            counter += 1
            labels.append(valLabels[i])
            valStorage.append(face[None])
    file.create_array(file.root, "valLabels", labels)
    labels = []
    for i in range(len(testData)):
        if i % 1000 == 0:
            print("test data : %d of %d" %(i, len(testData)))
        face = filterFace(testData[i])
        if face is not None:
            # print("We got a None for %s" %testLabels[i])
            counter += 1
            labels.append(testLabels[i])
            testStorage.append(face[None])
    file.create_array(file.root, "testLabels", labels)
    file.close()
    print(counter, len(dataLst))


def readFromHDF5(filePath, batchSize, selectNum):
    file = tables.open_file(filePath, mode='r')
    trainBatches, labels = readData(file.root.trainImg, file.root.trainLabels, batchSize, "train", selectNum)
    valBatches, labels = readData(file.root.valImg, file.root.valLabels, batchSize, "val", selectNum, labels)
    testBatches, labels = readData(file.root.testImg, file.root.testLabels, batchSize, "test", selectNum, labels)
    file.close()
    return trainBatches, testBatches, valBatches

def readData(images, labels, batchSize, dataType, selectNum, setLbls=None):
    dataNum = images.shape[0]
    imgs = []
    lbls = []
    counter = 0
    for i in range(dataNum):
        if i % 10000 == 0:
            print("added %d of %d" %(i, dataNum))
        if setLbls is None or labels[i] in setLbls:
            if labels[i] not in lbls:
                counter += 1
                if counter < selectNum:
                    imgs.append(images[i])
                    lbls.append(labels[i])
            else:
                imgs.append(images[i])
                lbls.append(labels[i])
    images, labels = shuffle(imgs, lbls)
    writer = tf.python_io.TFRecordWriter("%s.tfrecords" %dataType)
    i = 0
    counter = 0
    while i < len(images):
        firstImg = i
        finalImg = min([(i + batchSize), dataNum])
        img = images[firstImg: finalImg]
        label = labels[firstImg: finalImg]
        writeToTensor(writer, img, label, dataType)
        print(i, len(images))
        i = finalImg
        counter += 1
    writer.close()
    return counter, labels

def intTensor(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def byteTensor(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def writeToTensor(writer, images, labels, dataType):
    seenLabel = []
    for i in range(len(labels)):
        if labels[i] not in seenLabel:
            seenLabel.append(labels[i])
        labelVal = seenLabel.index(labels[i])
        feature = {"%s/label" %dataType : intTensor(labelVal), 
                   "%s/image" %dataType : byteTensor(tf.compat.as_bytes(images[i].tostring()))
                }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())





