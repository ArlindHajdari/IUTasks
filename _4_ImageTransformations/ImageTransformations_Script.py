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

def breakImage(image, blocks = 16):
    blocked_image = np.zeros((blocks, blocks), dtype=object)
    x_split_by_percentage = image.shape[0]//blocks
    y_split_by_percentage = image.shape[1]//blocks
    
    for i in range(0, image.shape[0] - x_split_by_percentage + 1, x_split_by_percentage):
        for j in range(0, image.shape[1] - y_split_by_percentage + 1, y_split_by_percentage):
            blocked_image[i//x_split_by_percentage,j//y_split_by_percentage] = image[i: i + blocks, j: j + blocks]

    return blocked_image

def lowResolution(image):
    x, y = image.shape
    low_resolution_image = np.zeros((x,y))

    for i in range(0, x):
        for j in range(0, y):
            low_resolution_image[i,j] = image[i,j].mean()

    return low_resolution_image

def main():
    # ============= Negative image ================
    img = mpimg.imread("../test_images/lena.png")
    img_negative = get_negative_img(img)

    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # fig.suptitle("Negativity")
    # plt.ion()
    # axis1.imshow(img, cmap = "gray")
    # axis2.imshow(img_negative, cmap = "gray")
    # plt.show(block=True)

    # ============= Mirroring image ===============
    # img = mpimg.imread("../test_images/lena.png")
    img_mirror = np.fliplr(img_negative)

    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # fig.suptitle("Mirroring")
    # plt.ion()
    # axis1.imshow(img, cmap = "gray")
    # axis2.imshow(img_mirror, cmap = "gray")
    # plt.show(block=True)

    # ============= Add/Substraction of image =====
    # img = mpimg.imread("../test_images/lena.png")
    random_value = randint(0,256)
    img_transformed_pixels = transorm_pixels(img_mirror, random_value)
    clipped_image = clip_img(img_transformed_pixels, 0, 255)

    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # fig.suptitle("Transforming Add/Substraction with value: "+ str(random_value))
    # plt.ion()
    # axis1.imshow(img, cmap = "gray")
    # axis2.imshow(clipped_image, cmap = "gray")
    # plt.show(block=True)

    # ============= Tear image into 16x16 blocks and compute histogram =====
    # img = mpimg.imread("../test_images/lena.png")
    block_size = 16
    teared_up_image = breakImage(img, block_size)
    teared_up_image_with_histograms = np.zeros((block_size, block_size), dtype=object)

    for i in range(0, block_size):
        for j in range(0, block_size):
            teared_up_image_with_histograms[i,j] = imhist(teared_up_image[i,j].flatten(), 256)

    fig, ax = plt.subplots(nrows=block_size, ncols=block_size)
    fig.suptitle(f"Tear image into {block_size}x{block_size} blocks and compute histogram")
    plt.ion()
    i = 0
    for row in ax:
        j = 0
        for col in row:
            col.plot(teared_up_image_with_histograms[i][j])
            j = j + 1
        i = i + 1
    
    plt.show(block=True)

    # ================= Low Resolution image =======================
    # img = mpimg.imread("../test_images/lena.png")
    # teared_up_image = breakImage(img, blocks = 16)
    low_resolution_image = lowResolution(teared_up_image)
    
    plt.ion()
    plt.imshow(low_resolution_image, cmap='gray')
    plt.show(block=True)


if __name__ == "__main__":
    main()