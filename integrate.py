import numpy as np


def integrateImage(img):
    intImage = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int)
    intImage[0:] = 0
    intImage[:0] = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            intImage[row + 1, col + 1] = intImage[row, col + 1] + intImage[row + 1, col] - intImage[row, col] + img[row, col]
    ret = intImage[1:, 1:]
    return ret


def regionArea(integral, topLeft, bottomRight):
    a = integral[topLeft[0], topLeft[1]]
    b = integral[topLeft[0], bottomRight[1]]
    c = integral[bottomRight[0], topLeft[1]]
    d = integral[bottomRight[0], bottomRight[1]]
    area = d - b - c + a
    return area
