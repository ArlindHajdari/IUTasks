import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plim


def imhist(image, bins):
    histogram = np.zeros(bins)
    
    for i in image: 
        index = int(i*255)
        histogram[index] += 1
    
    return histogram


def main():
    image = plim.imread("../test_images/barbara.png")
    array_image = np.asarray(image)
    flatten_image = array_image.flatten()
    hist = imhist(flatten_image,256)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Histogram')
    ax1.imshow(image, cmap='gray')
    ax2.plot(hist)
    plt.show(block=True)


if __name__ == "__main__":
    main()