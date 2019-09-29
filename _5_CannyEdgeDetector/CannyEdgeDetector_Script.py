import sys

sys.path.append('../_2_SpatialFiltering')

from SpatialFiltering_Script import convolution2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class cannyEdgeDetector:
    # Constructor
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img = img
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2 
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal # The equation for a Gaussian filter
        return g
    
    def sobel_filters(self, img):
        # Double filters to work on both axes
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = convolution2d(img, Kx, 2) # Apply convolution in X axis
        Iy = convolution2d(img, Ky, 2) # Apply convolution in Y axis

        G = np.hypot(Ix, Iy) # Equivalent to sqrt(Ix**2 + Iy**2)
        G = G / G.max() * 255 # G calculation
        theta = np.arctan2(Iy, Ix) # Theta calculation
        return (G, theta)
    

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi # Radian to degrees
        angle[angle < 0] += 180 # angels under 0 increase by 180

        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold # Get the highest threshold
        lowThreshold = highThreshold * self.lowThreshold # Get the lowest threshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold) # Fetch strongest pixels
        zeros_i, zeros_j = np.where(img < lowThreshold) # Fetch lowest pixels

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold)) # Pixels between low and high threshold

        res[strong_i, strong_j] = strong # Override the strong pixels
        res[weak_i, weak_j] = weak # Override the weak pixels

        return (res)

    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        
        # Transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
    
    def detect(self):  
        self.img_smoothed = convolution2d(self.img, self.gaussian_kernel(self.kernel_size, self.sigma), 3)
        printImage(self.img_smoothed, "Smoothed image")
        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        printImage(self.gradientMat, "Sobel filter applied")
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
        printImage(self.nonMaxImg, "Non max suppression")
        self.thresholdImg = self.threshold(self.nonMaxImg)
        printImage(self.thresholdImg, "Thresholded image")
        self.img = self.hysteresis(self.thresholdImg)
        return self.img

def printImage(img, description):
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle(description)
    ax1.imshow(img, cmap='gray')
    plt.show(block=True)

def main():
    img =  mpimg.imread("../test_images/lena.png") # Read the image

    detector = cannyEdgeDetector(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100) # Initialize with custom parameters
    final_img = detector.detect() # Apply the functions

    # Show original and transformed image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Canny edge detector')
    ax1.imshow(img, cmap='gray')
    ax2.imshow(final_img, cmap='gray')
    plt.show(block=True)


if __name__ == "__main__":
    main()