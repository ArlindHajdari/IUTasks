import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plim


def imhist(image, bins):
    histogram = np.zeros(bins) # Array of size of bins(usually 256)
    
    for i in image: 
        index = int(i*255) # Since image has values between 0 and 1, translate to 0 to 255
        histogram[index] += 1 # Count the pixels
    
    return histogram


def main():
    image = plim.imread("../test_images/barbara.png") # Read the image
    array_image = np.asarray(image) # Image to numpy array convertion
    
    
    flatten_image = array_image.flatten() # Numpy array to single list convertion
    hist = imhist(flatten_image,256) # Histogram computation
    
    # Show image with its histogram
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Histogram of the given image')
    ax1.imshow(image, cmap='gray')
    ax2.plot(hist)
    plt.show(block=True)


if __name__ == "__main__":
    main()