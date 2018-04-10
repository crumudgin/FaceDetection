from HaarDetection import *
import numpy as np
import math


def adaboost(training):
    n = len(training)
    # number of correct images in training data
    m = 5
    # number of incorrect images in training data
    l = 5
    # total number of training data
    n = m + l
    for imgIndex, img in enumerate(training):
        yi = 1
        features = allFeature(img)
        # training time
        T = 10
        d = np.array((T, n))
        d[1, imgIndex] = 1/m
        for t in range(T):

            for idx, j in enumerate(features):
                sum = 0
                h = []

                for i in range(m):
                    # appending error
                    h.append(d[t, i] * hx(j, img))
                # finding index of least error
                ht = h[np.argmin(h)]
                et = 0
                if et >= 0.5:
                    continue
                alphat = 0.5 * math.log((1-et)/et)
                # update
                #normalizationFactor
                zt =1
                d[t+1, imgIndex] = (d[t, imgIndex] * math.exp(- alphat * yi * hx(j, img))) / zt


def hx(feature, img):
    threshold = 5
    intImage = integrate.integrateImage(img)
    return 1 if feature.getAreaDiff(intImage) < threshold else -1
