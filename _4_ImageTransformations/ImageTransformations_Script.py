import sys

sys.path.append('../_3_Histogram')

from Histogram_Script import imhist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

def get_negative_img(image):
    x, y = image.shape
    negative = np.zeros((x, y))

    for i in range(0, x):
        for j in range(0, y):
            negative[i,j] = 255 - int(255 * image[i,j])

    return negative


def transorm_pixels(image, value, isAddition = False):
    x, y = image.shape
    transformed_image = np.zeros((x, y))

    for i in range(0, x):
        for j in range(0, y):
            transformed_image[i,j] = int(255 * image[i,j]) + value if isAddition else int(255 * image[i,j]) - value
    
    return transformed_image


def clip_img(image, minVal, maxVal):
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i,j] < minVal:
                image[i,j] += maxVal
            elif image[i,j] > maxVal:
                image[i,j] -= maxVal

    return image

def main():
    # ============= Negative image ================
    img = mpimg.imread("../test_images/lena.png")
    img_negative = get_negative_img(img)

    fig, (axis1, axis2) = plt.subplots(1, 2)
    fig.suptitle("Negativity")
    plt.ion()
    axis1.imshow(img, cmap = "gray")
    axis2.imshow(img_negative, cmap = "gray")
    plt.show(block=True)

    # ============= Mirroring image ===============
    # img = mpimg.imread("../test_images/lena.png")
    # img_mirror = np.fliplr(img)

    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # fig.suptitle("Mirroring")
    # plt.ion()
    # axis1.imshow(img, cmap = "gray")
    # axis2.imshow(img_mirror, cmap = "gray")
    # plt.show(block=True)

    # ============= Add/Substraction of image =====
    # img = mpimg.imread("../test_images/lena.png")
    # random_value = randint(0,256)
    # img_transformed_pixels = transorm_pixels(img, random_value)
    # clipped_image = clip_img(img_transformed_pixels, 0, 255)

    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # fig.suptitle("Transforming Add/Substraction with value: "+ str(random_value))
    # plt.ion()
    # axis1.imshow(img, cmap = "gray")
    # axis2.imshow(clipped_image, cmap = "gray")
    # plt.show(block=True)


if __name__ == "__main__":
    main()