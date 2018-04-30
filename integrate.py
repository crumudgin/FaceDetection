import numpy as np

def integrateImage(img):
    """
    find integrate image of the image pass for haar and classification.
    :param img: image to find intergrate of
    :return: integrate version of the image
    """
    intImage = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int)
    intImage[0:] = 0
    intImage[:0] = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # integrate dynamic programming code
            intImage[row + 1, col + 1] = intImage[row, col + 1] + intImage[row + 1, col] - intImage[row, col] + img[row, col]
    ret = intImage[1:, 1:]
    return ret


def regionArea(integral, topLeft, bottomRight):
    """
    gives area of the given coordinate found
    :param integral: integrate version of the image
    :param topLeft: top left coordinate of the rectangle
    :param bottomRight: bottom right coordinate of the rectangle
    :return: area of the given rectangle
    """
    a = integral[topLeft[0], topLeft[1]]
    b = integral[topLeft[0], bottomRight[1]]
    c = integral[bottomRight[0], topLeft[1]]
    d = integral[bottomRight[0], bottomRight[1]]
    area = d - b - c + a
    return area

# debug code
# img = np.array((24,24))
# integrat = integrateImage(img)
#
# print(integrat)