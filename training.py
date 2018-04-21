import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import os
import sys
import cv2
from os.path import isfile, join, isdir

#
# def main():
#     if (len(sys.argv) < 2):
#         print('need person\'s name')
#         print("Usage: python training.py FirstName_LastName")
#         sys.exit()
#
#     side = cv2.CascadeClassifier("sideFace.xml")
#     front = cv2.CascadeClassifier("FrontalFace.xml")
#     cam = cv2.VideoCapture(0)
#
#     personName = str(sys.argv[1])
#     picCount= 0
#     while(True):
#         ret, img = cam.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         rightside = side.detectMultiScale(gray, 1.2,5)
#         horizontal_img = cv2.flip(gray, 1)
#         leftside = side.detectMultiScale(horizontal_img, 1.2, 5)
#         frontFace = front.detectMultiScale(gray, 1.2, 5)
#         directory = "trainer/" + personName + str("/")
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#
#         for (x,y, w, h) in frontFace:
#             face = img[y:y + h, x: x + w]
#             face = aligment(face)
#             cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
#             picCount += 1
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         for (x, y, w, h) in rightside:
#             face = img[y:y + h, x: x + w]
#             face = aligment(face)
#             cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
#             picCount += 1
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         for (x, y, w, h) in leftside:
#             face = img[y:y + h, x: x + w]
#             face = aligment(face)
#             cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
#             picCount += 1
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         cv2.imshow("faces", img)
#         if(cv2.waitKey(1) == ord('q')):
#             break
#     cam.release()
#     cv2.destroyAllWindows()


def aligment(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 2)

    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        #(x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, image, rect)
        return faceAligned

def readImageFromFolder(name):
    givenPath=[name]
    images = []
    for path in givenPath:
        for f in os.listdir(path):
            if isfile(join(path,f)):
                images.append(join(path,f))
            elif isdir(join(path,f)):
                givenPath.append(join(path,f))
    return images


def main():
    images = readImageFromFolder("face")
    count = 0
    for img in images:
        name = img.split("\\")[1]
        imgName = img.split("\\")[2]
        directory = "trainer/aligned/" + name + str("/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        img = cv2.imread(img)
        alignedFace = aligment(img)
        cv2.imwrite(directory + imgName, alignedFace)
        count += 1


main()
