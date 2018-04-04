from __future__ import print_function
import numpy as np
from enum import Enum
import cv2


class Features(Enum):
    # 1,2 means in two vertical divide row in one , divide col in two
    #      (2,1)           (2,2)          (1,2)             (1,3)
    #    ________        ________        ________          _________
    #   |        |      |    ####|      |    ####|        |   ###   |
    #   |        |      |    ####|      |    ####|        |   ###   |
    #   |########|      |####    |      |    ####|        |   ###   |
    #   |########|      |####    |      |    ####|        |   ###   |
    #   ----------       --------        --------          ---------
    #   2 horizontal    4 -vertical     2-vertical
    TWO_VERTICAL = [1, 2]
    TWO_HORIZONTAL = [2, 1]
    THREE_HORIZONTAL = [3, 1]
    THREE_VERTICAL = [1, 3]
    FOUR_VERTICAL = [2, 2]


def integrateImage(img):
    intergralImage = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int)
    intergralImage[0:] = 0
    intergralImage[:0] = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            intergralImage[row + 1, col + 1] = intergralImage[row, col + 1] + intergralImage[row + 1, col] - intergralImage[row, col] + img[row, col]
    ret = intergralImage[1:, 1:]
    return ret


def regionArea(integral, topLeft, bottomRight):
    a = integral[topLeft[0], topLeft[1]]
    b = integral[topLeft[0], bottomRight[1]]
    c = integral[bottomRight[0], topLeft[1]]
    d = integral[bottomRight[0], bottomRight[1]]
    area = d - b - c + a
    return area


def allFeature(img):
    allFeatures = []
    for feature in Features:
        minWidth = max(1, feature.value[0])
        minHeight = max(1, feature.value[1])
        for featureWidth in range(minWidth, img.shape[0], feature[0]):
            for featureHeight in range(minHeight, img.shape[1], feature[1]):
                for x in range(0, img.shape[0] - featureWidth):
                    for y in range(0, img.shape[1] - featureHeight):
                        print ("I am new feature")


img = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
integtal = integrateImage(img)

for row in range(integtal.shape[0]):
    for col in range(integtal.shape[1]):
        print(integtal[row, col], end="\t")
    print()

a = regionArea(integtal, (1, 1), (3, 3))
print(a)

for shake in Features:
    print(str(shake.value[0]) +str(",") + str(shake.value[1]))



