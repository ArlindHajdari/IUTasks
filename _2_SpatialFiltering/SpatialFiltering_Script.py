import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def convolution2d(image, kernel, pad):
    m, n = kernel.shape
    image = pad_image(image, pad)
    y, x = image.shape
    y_out = y - m + 1
    x_out = x - n + 1
    new_image = np.zeros((y_out, x_out))

    for i in range(y_out):
        for j in range(x_out):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n]*kernel)

    return new_image

def pad_image(image, pad):
    m, n = image.shape
    axes_change = pad * 2
    padded_image = np.zeros((m+axes_change, n+axes_change))

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            padded_image[i+pad,j+pad] = image[i,j]
    
    return padded_image

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
