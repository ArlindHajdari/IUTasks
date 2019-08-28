import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def convolution2d(image, kernel, pad):
    m, n = kernel.shape # Get the shape of the kernel
    y, x = image.shape # Get the shape of the image

    image = pad_image(image, pad) # Pad the image for the given pad
    
    x_out = x - n + 1 # Create the new value for the x axis
    y_out = y - m + 1 # Create the new value for the y axis
   
    new_image = np.zeros((x_out, y_out)) # Create filled with zeros matrix of size x_out x y_out

    for i in range(x_out):
        for j in range(y_out):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel) # Compute the sum of the multiplied matrixes (pattern of the image of size of the kernel, and kernel itself)

    return new_image

def pad_image(image, pad):
    m, n = image.shape
    axes_change = pad * 2 # Since the axes will change twice the size of the pad (left - right, bottom - top)
    padded_image = np.zeros((m+axes_change, n+axes_change)) # Create an empty (filled with zeros) matrix of the new dimensions

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            padded_image[i+pad,j+pad] = image[i,j]
    
    return padded_image

def main():
    image = mpimg.imread('../test_images/lena.png') # Read the image from the path
    kernel = np.array([[1, 2, 1], [1, -2, 1], [-1, 2, -1]]) 
    new_image = convolution2d(image, kernel, 2) # Compute the convolution of the image by kernel with padding = 2 (kernel size - 1)
    
    # Show the image
    figure, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(image, cmap = "gray")
    axis2.imshow(new_image, cmap = "gray")
    plt.show(block=True)

if __name__ == '__main__':
    main()
