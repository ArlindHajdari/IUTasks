import sys

sys.path.append('../_3_Histogram')

from Histogram_Script import imhist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint


def get_negative_img(image):
    x, y = image.shape # Size of the image
    negative = np.zeros((x, y)) # Create an matrix of the same size filled with zeros

    # For every pixel of the image subtract 255
    for i in range(0, x):
        for j in range(0, y):
            negative[i,j] = 255 - int(255 * image[i,j]) # Since image has values between 0 and 1, this multiplication is done to translate to the range between 0 and 255

    return negative


def transorm_pixels(image, value, isAddition = False):
    x, y = image.shape
    transformed_image = np.zeros((x, y))

    # Depending on isAddition parameter add/subtract value from every pixel in the image
    for i in range(0, x):
        for j in range(0, y):
            transformed_image[i,j] = int(255 * image[i,j]) + value if isAddition else int(255 * image[i,j]) - value
    
    return transformed_image


def clip_img(image, minVal, maxVal):
    # Clip every pixel in the image to be between minVal and maxVal

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i,j] < minVal: # If the current pixel is less than minVal 
                image[i,j] += maxVal # Add the minVal
            elif image[i,j] > maxVal: # else if the current pixel is greater than maxVal
                image[i,j] -= maxVal # Subtract the maxVal

    return image


def breakImage(image, blocks = 16):
    blocked_image = np.zeros((blocks, blocks), dtype=object) # Initialization
    x_split_by_percentage = image.shape[0]//blocks # Get the size of x axis
    y_split_by_percentage = image.shape[1]//blocks # Get the size of y axis
    
    # Looping to the image from 0 to maxVal of x axis minus the split value, the counter increases by split value
    for i in range(0, image.shape[0] - x_split_by_percentage + 1, x_split_by_percentage):
        for j in range(0, image.shape[1] - y_split_by_percentage + 1, y_split_by_percentage):
            blocked_image[i//x_split_by_percentage,j//y_split_by_percentage] = image[i: i + blocks, j: j + blocks]

    return blocked_image


def lowResolution(image):
    x, y = image.shape
    low_resolution_image = np.zeros((x,y))

    for i in range(0, x):
        for j in range(0, y):
            low_resolution_image[i,j] = int(255 * image[i,j].mean()) # Since image has 0 to 1 values we need to do the multiplication

    return low_resolution_image


def applyThreshold(image, threshold):
    x, y = image.shape
    thresholdedImage = np.zeros((x, y))

    # Pixels above the threshold are replaced by 0
    for i in range(0, x):
        for j in range(0, y):
            if image[i,j] > threshold:
                thresholdedImage[i, j] = 0 
            else:
                thresholdedImage[i, j] = image[i,j]

    return thresholdedImage


def main():
    img = mpimg.imread("../test_images/lena.png") # Read image from the path
    
    # ============= Negative image ================
    img_negative = get_negative_img(img) #Lighest values appear dark and vice versa

    # Show the images (new and computed)
    fig, (axe1, axe2) = plt.subplots(1,2)
    fig.suptitle("Negative image")
    axe1.imshow(img, cmap = "gray")
    axe2.imshow(img_negative, cmap = "gray")
    plt.show()

    # ============= Mirroring image ===============
    img_mirror = np.fliplr(img) #flipping image left to right

    fig, (axe1, axe2) = plt.subplots(1,2)
    fig.suptitle("Flipped image left to right")
    axe1.imshow(img, cmap = "gray")
    axe2.imshow(img_mirror, cmap = "gray")
    plt.show()

    # ============= Add/Substraction of image =====
    random_value = randint(0,256) #generate a random value between 0, 256 (doesn't get the higher limit)
    img_transformed_pixels = transorm_pixels(img, random_value) #by default add, random_value to everypixel in the image
    clipped_image = clip_img(img_transformed_pixels, 0, 255) #clip image to have minimum 0 and maximum 255
    
    fig, (axe1, axe2) = plt.subplots(1,2)
    fig.suptitle(f"Add/Substraction and Clipped image by {random_value}")
    axe1.imshow(img, cmap = "gray")
    axe2.imshow(clipped_image, cmap = "gray")
    plt.show()

    # ============= Tear image into 16x16 blocks and compute histogram =====
    block_size = 16 # blocks to tear the image
    teared_up_image = breakImage(img, block_size) # Tear image into block_size x block_size non overlapping blocks
    teared_up_image_with_histograms = np.zeros((block_size, block_size), dtype=object) # Create an empty(with zeros) matrix of size block_size x block_size

    # For every pixel of the teared_up_image_with histograms compute the histogram of the corresponding pixel at teared_up_image
    for i in range(0, block_size):
        for j in range(0, block_size):
            teared_up_image_with_histograms[i,j] = imhist(teared_up_image[i,j].flatten(), 256)

    fig, ax = plt.subplots(nrows=block_size, ncols=block_size)
    fig.suptitle(f"Tear image into {block_size}x{block_size} blocks and compute histogram")
    
    i = 0
    for row in ax:
        j = 0
        for col in row:
            col.axis('off')
            col.plot(teared_up_image_with_histograms[i][j])
            j = j + 1
        i = i + 1
    
    plt.show(block=True)

    # ================= Low Resolution image =======================
    low_resolution_image = lowResolution(teared_up_image) # For each of the blocks, compute mean value
    
    fig, (axe1, axe2) = plt.subplots(1,2)
    fig.suptitle("Low resolution image")
    axe1.imshow(img, cmap = 'gray')
    axe2.imshow(low_resolution_image, cmap = 'gray')
    plt.show(block=True)

    # ================= Segment the face of Lena ===================
    while True:
        threshold = input("Type the threshold(0 to 255)<type exit to break>: ") # Get the threshold from the client

        # If client doesn't want to exit apply the threshold to the image
        if "exit" not in threshold:
            thresholded_image = applyThreshold(low_resolution_image, int(threshold)) # Applying the threshold

            # Show the calculated results
            fig, (axe1, axe2) = plt.subplots(1,2)
            fig.suptitle(f"Thresholded image by {int(threshold)}")
            axe1.imshow(low_resolution_image, cmap = 'gray')
            axe2.imshow(thresholded_image, cmap = 'gray')
            plt.show(block=True)
        else:
            break

# Call the main by deafult
if __name__ == "__main__":
    main()