from __future__ import print_function
import numpy as np
from enum import Enum
import integrate
import cv2


class Features(Enum):
    # 1,2 means in two vertical divide row in one , divide col in two
    #      (2,1)         (2,2)       (1,2)        (1,3)          (3,1)
    #    ________      ________     ________     _________     __________
    #   |        |    |    ####|   |    ####|   |   ###   |   |          |
    #   |        |    |    ####|   |    ####|   |   ###   |   |##########|
    #   |########|    |####    |   |    ####|   |   ###   |   |##########|
    #   |########|    |####    |   |    ####|   |   ###   |   |          |
    #   ----------     --------     --------     ---------     ----------
    #  2 horizontal   4-vertical   2-vertical    3-vertical   3-horizontal
    TWO_VERTICAL = [1, 2]
    TWO_HORIZONTAL = [2, 1]
    THREE_HORIZONTAL = [3, 1]
    THREE_VERTICAL = [1, 3]
    FOUR_VERTICAL = [2, 2]

class Haar(object):
    def __init__(self, topLeft, bottomRight, feature):
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.feature = feature

    def getAreaDiff(self, intImage):
        if self.feature == Features.TWO_HORIZONTAL:
            blackRegion = integrate.regionArea(intImage, self.topLeft, (self.bottomRight[0]/2, self.bottomRight[1]))
            whiteRegion = integrate.regionArea(intImage, (self.bottomRight[0]/2, self.bottomRight[1]), self.bottomRight)
            return blackRegion - whiteRegion
        elif self.feature == Features.TWO_VERTICAL:
            blackRegion = integrate.regionArea(intImage, (self.topLeft[0], self.bottomRight[1]/2), (self.bottomRight[0], self.bottomRight[1]/2))
            whiteRegion = integrate.regionArea(intImage, (self.topLeft[0], self.bottomRight[1]), self.bottomRight)
            return blackRegion - whiteRegion
        elif self.feature == Features.THREE_HORIZONTAL:
            firstRegion = integrate.regionArea(intImage, self.topLeft, (self.bottomRight[0] / 3, self.bottomRight[1]))
            secondRegion = integrate.regionArea(intImage, (self.bottomRight[0] / 3, self.topLeft[1]), (self.bottomRight[0] / 3 * 2, self.bottomRight[1]))
            thirdRegion = integrate.regionArea(intImage, (self.bottomRight[0] / 3 * 2, self.topLeft[1]), self.bottomRight)
            return secondRegion - firstRegion - thirdRegion
        elif self.feature == Features.THREE_VERTICAL:
            firstRegion = integrate.regionArea(intImage, self.topLeft, (self.bottomRight[0], self.bottomRight[1] / 3))
            secondRegion = integrate.regionArea(intImage, (self.topLeft[1], self.bottomRight[1] / 3), (self.bottomRight[0], self.bottomRight[1] / 3 * 2))
            thirdRegion = integrate.regionArea(intImage, (self.topLeft[1], self.bottomRight[1] / 3 * 2), self.bottomRight)
            return secondRegion - firstRegion - thirdRegion
        elif self.feature == Features.FOUR_VERTICAL:
            firstRegion = integrate.regionArea(intImage, self.topLeft, (self.bottomRight[0] / 2, self.bottomRight[1] / 2))
            secondRegion = integrate.regionArea(intImage, (self.topLeft[0], self.bottomRight[1] / 2), (self.bottomRight[0] / 2, self.bottomRight[1]))
            thirdRegion = integrate.regionArea(intImage, (self.bottomRight[0] / 2, self.topLeft[0]), (self.bottomRight[0], self.bottomRight[1] / 2))
            fourthRegion = integrate.regionArea(intImage, (self.bottomRight[0] / 2, self.bottomRight[1] / 2), self.bottomRight)
            return firstRegion + fourthRegion - secondRegion - thirdRegion


def allFeature(img):
    allFeatures = []
    count = 0
    # for each feature
    for feature in Features:
        # minimum size to start feature
        minWidth = max(1, feature.value[0])
        minHeight = max(1, feature.value[1])
        for featureWidth in range(minWidth, img.shape[0]+1, minWidth):
            for featureHeight in range(minHeight, img.shape[1]+1, minHeight):
                for x in range(0, img.shape[0] - featureWidth+1):
                    for y in range(0, img.shape[1] - featureHeight+1):
                        allFeatures.append(Haar((x, y), (x+featureHeight, y+featureWidth), feature))
    return allFeatures


img = np.ones((24, 24))

c = allFeature(img)

# fea = list(range(len(c)

# topLeft = (0,0)
# bottomRight = (10,10)

# img = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# integtal = integrateImage(img)
#
#
#
# for row in range(integtal.shape[0]):
#     for col in range(integtal.shape[1]):
#         print(integtal[row, col], end="\t")
#     print()
#
# a = regionArea(integtal, (1, 1), (3, 3))
# print(a)
#
# for shake in Features:
#     print(str(shake.value[0]) + str(",") + str(shake.value[1]))
#


