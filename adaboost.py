from HaarDetection import *
import numpy as np
import math

def readImageFromFolder(name):
    """
    read all image from the given folder location
    :param name: name of the folder
    :return: images location of all images
    """
    givenPath=[name]
    images = []
    # for all path in given path
    for path in givenPath:
        # if the given path is directory add in directory to read from
        for f in listdir(path):
            # if is a file it read it.
            if isfile(join(path,f)):
                # append that image file in the list of images
                images.append(join(path,f))
                # if it is a directory add that location to the givenpath to read from it later
            elif isdir(join(path,f)):
                givenPath.append(join(path,f))
    # returns list of all image directory.
    return images


def findBestWeakClassifier(samples, nSamples, weak):
    """
    implemenation if adaboost foread from the face images
    :param samples: sample of images
    :param nSamples: number of samples
    :param weak: weak classifier
    :return: returns classifier which can correctly detect face.
    """

    best_error = 111111111.1
    best_index = 0
    best_threshold = 1
    best_parity = 1
    numFeatures = 0
    # for number of feature
    for n in range(numFeatures):
        samplePtr = []
        # for number of samples
        for s in range(nSamples):
            samplePtr[s] = samples[s]
        # sort sample
        samplePtr.sort()

        sum_pos = 0.0
        sum_pos_below = 0.0
        sum_neg = 0.0
        sum_neg_below = 0.0
        # for number of sample we have
        for s in range(nSamples):
            # if sample is positive and is face
            if samples[s].label == 1:
                # increase positive weight
                sum_pos += samples[s].weight
            else:
                # increase negative weight
                sum_neg += samples[s].weight

        prev_sum_pos = 0
        prev_sum_neg = 0
        # for number of samples
        for s in range(nSamples):
            # get feature of samples
            threshold = samplePtr[s].features[n]
            if s > 0:
                if samplePtr[s-1].features[n] == threshold:
                    if samplePtr[s-1].label == 1:
                        prev_sum_pos += samplePtr[s-1].weight
                    else:
                        prev_sum_neg += samplePtr[s-1].weight
                else:
                    sum_pos_below += prev_sum_pos
                    sum_neg_below += prev_sum_neg
                    if samplePtr[s-1].label == 1:
                        sum_pos_below += samplePtr[s-1].weight
                    else:
                        sum_neg_below += samplePtr[s-1].weight
                    prev_sum_pos, prev_sum_neg = 0

            parity_error = []
            parity_error[0] = sum_neg_below + (sum_pos - sum_pos_below)
            parity_error[1] = sum_pos_below + (sum_neg - sum_pos_below)
            # error = 0.0
            # parity = 1
            if parity_error[0] < parity_error[1]:
                error = parity_error[0]
                parity = 0
            else:
                error = parity_error[1]
                parity = 1

            if error < best_error:
                best_index = n
                best_error = error
                best_threshold = threshold
                best_parity = parity
                print(" new best feature found")
    # change weak classifier property.

    weak.index = best_index
    weak.threshold = best_threshold
    weak.parity = best_parity
    weak.alpha = math.log((1-best_error)/best_error)/2



#
# def adaboost():
#     trainingImg = []
#     count = 0
#     # all images
#     faces = readImageFromFolder("/faces/")
#     notfaces = readImageFromFolder("/notdFaces/")
#     # all non faces images
#     trainingImg.append(faces)
#     trainingImg.append(notfaces)
#     y = []
#     y[0:len(faces)] = 1
#     y[len(faces): len(faces) + len(notfaces)] = -1
#     # number of correct images in training data
#     m = len(faces)
#     # number of incorrect images in training data
#     l = len(notfaces)
#     # total number of training data
#     n = m + l
#     alpha = []
#     h = []
#
#     T = 10
#     # for t = 1, ..., T
#     for t in range(T):
#
#         features = allFeature(img)
#         # training time
#         d = np.array((T, n))
#         # for each image
#         for imgIndex, img in enumerate(trainingImg):
#             d[t, imgIndex] = 1 / m
#             # for each feature
#             for idx, j in enumerate(features):
#                 # calculate sum and find Ej
#                 for i in range(m):
#
#                     # appending error
#                     h.append(d[t, i] * hx(j, img))
#                 # finding index of least error
#                 ht = h[np.argmin(h)]
#
#                 et = 0
#                 if et >= 0.5:
#                     continue
#                 alpha[t] = 0.5 * math.log((1-et)/et)
#                 # update
#                 #normalizationFactor
#                 zt =1
#                 d[t+1, imgIndex] = (d[t, imgIndex] * math.exp(- alpha[t] * y[imgIndex] * hx(j, img))) / zt
#     x = []
#     for t in range(T):
#         HX = alpha[t] * hx()
#
# """
#     this function finds best threshold for face. if area of the feature is less than the threshold
#     of the image then polarity is -1 or not face else 1
# """
# def hx(feature, img):
#     threshold = 5
#     polarity = 5
#     intImage = integrate.integrateImage(img)
#     return 1 if feature.getAreaDiff(intImage) < threshold * polarity else -1
