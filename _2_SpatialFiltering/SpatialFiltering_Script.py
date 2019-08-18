import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def convolution2d(image, kernel, pad):
    m, n = kernel.shape
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y, x = image.shape
    y_out = y - m + 1
    x_out = x - n + 1
    new_image = np.zeros((y_out, x_out))

    for i in range(y_out):
        for j in range(x_out):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n]*kernel)

    return new_image

def main():
    image = mpimg.imread('../test_images/lena.png')
    kernel = np.array([[1, 2, 1], [1, -2, 1], [-1, 2, -1]])
    new_image = convolution2d(image, kernel, 2)
    
    figure, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(image, cmap = "gray")
    axis2.imshow(new_image, cmap = "gray")
    plt.ion()
    plt.show(block=True)

if __name__ == '__main__':
    main()
