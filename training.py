import cv2
import numpy as np
import os
import sys

def main():
    if (len(sys.argv) < 2):
        print('need person\'s name')
        print("Usage: python training.py FirstName_LastName")
        sys.exit()

    side = cv2.CascadeClassifier("sideFace.xml")
    front = cv2.CascadeClassifier("FrontalFace.xml")
    cam = cv2.VideoCapture(0)
    personName = str(sys.argv[1])
    picCount= 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rightside = side.detectMultiScale(gray, 1.2,5)
        horizontal_img = cv2.flip(gray, 1)
        leftside = side.detectMultiScale(horizontal_img, 1.2, 5)
        frontFace = front.detectMultiScale(gray, 1.2, 5)

        directory = "trainer/" + personName + str("/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for (x,y, w, h) in frontFace:
            face = img[y:y + h, x: x + w]
            cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
            picCount += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in rightside:
            face = img[y:y + h, x: x + w]
            cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
            picCount += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in leftside:
            face = img[y:y + h, x: x + w]
            cv2.imwrite(directory + personName + str(picCount) + ".jpg", face)
            picCount += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("faces", img)
        if(cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()

main()
