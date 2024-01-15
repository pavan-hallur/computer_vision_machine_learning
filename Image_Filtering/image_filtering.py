#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.util import random_noise

from utils import *

#
# Task 1
#
#  Implement the the following filter functions such that each implements the respective image filter.
#  They shall not modify the input image, but return a filtered copy.
#  Implement at least one of them without using an existing filter function,
#  e.g. do not use the corresponding OpenCV functions `cv2._____Blur(..)`.

"""
Image smoothing:
1. Box filter
2. Sinc filter
3. Gauss filter
4. Median filter
"""


def filter_box(img, ksize=5, manual=True):
    if manual:
        # Create img to write output to.
        output = np.zeros(img.shape, dtype=img.dtype)

        # Create kernel to take the pixel average.
        kernel = 1/(ksize*ksize) * np.ones((ksize, ksize), dtype="float32")

        # Slide window over the image and apply the kernel for each channel at a time.
        rows, cols, channels = img.shape
        for channel in range(channels):
            for row in range(rows - ksize):
                for col in range(cols - ksize):
                    # Current window to apply the kernel to.
                    window = img[row:row+ksize, col:col+ksize, channel]
                    # Kernel convolution with current window.
                    kernel_convolution = window * kernel
                    # Set pixel average to output image.
                    output[row+int(ksize/2), col+int(ksize/2), channel] = np.sum(kernel_convolution)
        return output
    else:
        return cv2.boxFilter(img, -1, (ksize, ksize))


def filter_sinc(img, mask_circle_diameter=0.4):
    # Convert image to 1 channel (Grayscale).
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Transform from spatial to frequency domain.
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift zero frequencies to the center to analyse it easier.
    dft_shifted = np.fft.fftshift(dft)

    # Filter image in frequency domain.
    rows, cols = gray_img.shape
    center_row = rows // 2
    center_col = cols // 2
    # Create a mask to filter the image. (Needs 2 channels for real and imaginary part)
    mask = np.zeros((rows, cols, 2), dtype="uint8")
    radius = int(mask_circle_diameter / 2 * min(rows, cols))
    cv2.circle(mask, (center_col, center_row), radius, (1, 1), cv2.FILLED)
    # Apply mask to the frequency image.
    masked_dft = dft_shifted * mask
    # Invert shift.
    masked_dft_inverted_shift = np.fft.ifftshift(masked_dft)
    # Invert from frequency to spatial domain.
    masked_dft_inverted = cv2.idft(masked_dft_inverted_shift)
    # Convert from complex (2 channel) to 1 channel image.
    output = cv2.magnitude(masked_dft_inverted[:, :, 0], masked_dft_inverted[:, :, 1])

    # Debug: Find the magnitude spectrum from the frequency transform. (Normalize for cv2 display)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
    min_value   = np.min(magnitude_spectrum)
    value_range = np.ptp(magnitude_spectrum)
    magnitude_spectrum = (magnitude_spectrum - min_value) / value_range
    cv2.imshow("magnitude spectrum", magnitude_spectrum)

    # Debug: Show mask image.
    cv2.imshow("mask", mask[:, :, 0].astype("float32"))

    # Debug: Show masked magnitude_spectrum
    debug_masked_magnitude_spectrum = magnitude_spectrum * mask[:, :, 0]
    cv2.imshow("masked magnitude spectrum", debug_masked_magnitude_spectrum)

    return output


def filter_gauss(img, ksize = 5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def filter_median(img, ksize = 5):
    return cv2.medianBlur(img, ksize)


#
# (Task 2) 
#

"""
Implement another noise filtering / smoothing algorithm:
1. Bilateral filter
"""

def filter_bilateral(img, ksize=5):
    return cv2.bilateralFilter(img, ksize, 75, 75)


def applyFilter(filter, img):
    return globals()["filter_" + filter](img)


img1 = cv2.imread("img/geometric_shapes.png")

# Simulate image noise
noise_types = ["gaussian", "poisson", "s&p"]
imgs_noise = [from0_1to0_255asUint8(random_noise(img1, mode=n)) for n in noise_types]

imgs = [("original", img1)] + [(noise + " noise", img) for noise, img in zip(noise_types, imgs_noise)]
plt.figure(figsize=(10, 3))
showImages(imgs)

# Filter noise images
filter_types = ["box", "sinc", "gauss", "median", "bilateral"] # , "XYZ"] # (Task 2)
imgs_noise_filtered = [(f, [(noise, applyFilter(f, img)) for noise, img in imgs]) for f in filter_types]

imgs = imgs + [(f + " filter" if noise == "original" else "", img) for f, imgs_noise in imgs_noise_filtered for noise, img in imgs_noise]
plt.figure(figsize=(15, 8))
showImages(imgs, 4, transpose = True)


#
# Task 3
#

"""
Image brightness:
1. Reduce brightness of the input image.
2. Restore the bridghtness of the darkened image.
3. Explain with your own words why the "restored" picture shows that much noise.
4. Restore image with an average image of stacked noisy images.
"""

#  Simulate a picture captured in low light without noise.
#  Reduce the brightness of `img` about the provided darkening `factor`.
#  The data type of the returned image shall be the same as that of the input image.
#  Example (factor = 3): three times darker, i.e. a third of the original intensity.
def reduceBrightness(img, factor):
    # Change lightning in the HSV color space.
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:,:,2] = hsv_img[:,:,2] / factor
    # Convert HSV back to BGR.
    dark_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return dark_img


#  "Restore" the brightness of a picture captured in low light, ignoring potential noise.
#  Apply the inverse operation to `reduceBrightness(..)`.
def restoreBrightness(img, factor):
    # Change lightning in the HSV color space.
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:,:,2] = hsv_img[:,:,2] * factor
    # Clip max values to be 255.
    hsv_img[:,:,2] = np.where(hsv_img[:,:,2] > 255, 255, hsv_img[:,:,2])
    # Convert HSV back to BGR.
    bright_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return bright_img


img2 = cv2.imread("img/couch.jpg")
imgs = [("Original", img2)]

# Reduce image brightness
darkening_factor = 3
img_dark = reduceBrightness(img2, darkening_factor)

# Restore image brightness
img_restored = restoreBrightness(img_dark, darkening_factor)

imgs = imgs + [("Low light", img_dark), ("Low light restored", img_restored)]


# Simulate multiple pictures captured in low light with noise.
num_dark_noise_imgs = 10
imgs_dark_noise = [from0_1to0_255asUint8(random_noise(img_dark, mode="poisson")) for _ in range(num_dark_noise_imgs)]

# Now try to "restore" a picture captured in low light with noise (`img_dark_noise`) using the same function as for the picture without noise.
img_dark_noise = imgs_dark_noise[0]
img_noise_restored_simple = restoreBrightness(img_dark_noise, darkening_factor)

imgs = imgs + [None, ("Low light with noise", img_dark_noise), ("Low light with noise restored", img_noise_restored_simple)]


# Explain with your own words why the "restored" picture shows that much noise,
# i.e. why the intensity of the noise in low light images is typically so high compared to the image signal.
'''
We change the brightness by converting BGR to HSV color space.
In HSV we can simply reduce the V value (brightness).
When we restore the brightness, we simply increase the V value again.
But when convert it back to BGR, not only the brightness but everything is changed.
When converting it back to BGR we have multiplied the noise by a factor as well. 
________________________________________________________________________________
'''


#  Restore a picture from all the low light pictures with noise (`imgs_dark_noise`) by computing the "average image" of them.
#  Adjust the resulting brightness to the original image (using the `darkening_factor` and `num_dark_noise_imgs`).
img_noise_stack_restored = np.zeros(imgs_dark_noise[0].shape, dtype=imgs_dark_noise[0].dtype)
for img in imgs_dark_noise:
    # Compute the average restored image of all stacked images.
    img_noise_stack_restored = img_noise_stack_restored + restoreBrightness(img, darkening_factor) / num_dark_noise_imgs
img_noise_stack_restored = img_noise_stack_restored.astype("uint8") # We divided but want an uint image.


imgs = imgs + [("Low light with noise 1 ...", imgs_dark_noise[0]),
               ("... Low light with noise " + str(num_dark_noise_imgs), imgs_dark_noise[-1]),
               ("Low light stack with noise restored", img_noise_stack_restored)]
plt.figure(figsize=(15, 8))
showImages(imgs, 3)


#
# Task 4
#

"""
Edge detection:
1. Sobel filter in x and y direction
2. Take absolute values
3. Combine x and y
4. Normalize the image to be between [0,1]

5. Apply threshold to the sobel filtered image.
6. Apply the mask to the color image and set masked values to white.
"""

#
# Task 5
#

"""
Implement another edte detection algorithm:
1. Canny edge detector
"""

def filter_sobel(img, ksize = 3):
    #  Implement a sobel filter (x/horizontal + y/vertical) for the provided `img` with kernel size `ksize`.
    #  The values of the final (combined) image shall be normalized to the range [0, 1].
    #  Return the final result along with the two intermediate images.

    sobel_x = np.zeros(img.shape, dtype="float32")
    sobel_y = np.zeros(img.shape, dtype="float32")
    sobel   = np.zeros(img.shape, dtype="float32")

    # Create a copy of the image.
    copied_img = cv2.copyTo(img, mask=None)

    kernel_x = np.array([
        [+1, 0, -1],
        [+2, 0, -2],
        [+1, 0, -1]], dtype="float32")

    kernel_y = np.array([
        [+1, +2, +1],
        [ 0,  0,  0],
        [-1, -2, -1]], dtype="float32")
    
    rows, cols = copied_img.shape
    for col in range(cols-2):
        for row in range(rows-2):
            # Get the current window to use the kernel at.
            current_window = copied_img[row:row+3, col:col+3]
            # Compute center value of the window by multiplying with the kernel.
            sobel_x[row+1, col+1] = np.absolute( np.sum(current_window * kernel_x) )
            sobel_y[row+1, col+1] = np.absolute( np.sum(current_window * kernel_y) )

    #sobel = sobel_x + sobel_y
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize images to be between [0,1]: x = (x - xmin)/(xmax - xmin)
    min_value = np.min(sobel)
    max_value = np.max(sobel)
    sobel = (sobel - min_value) / (max_value - min_value)
    
    min_value = np.min(sobel_x)
    max_value = np.max(sobel_x)
    sobel_x = (sobel_x - min_value) / (max_value - min_value)
    
    min_value = np.min(sobel_y)
    max_value = np.max(sobel_y)
    sobel_y = (sobel_y - min_value) / (max_value - min_value)

    return sobel, sobel_x, sobel_y


def filter_canny(img):
    #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, 100, 200)


def applyThreshold(img, threshold):
    # Return an image whose values are 1 where the `img` values are > `threshold` and 0 otherwise.
    return np.where(img > threshold, 1, 0)


def applyMask(img, mask):
    #  Apply white color to the masked pixels, i.e. return an image whose values are 1 where `mask` values are 1 and unchanged otherwise.
    #  (All mask values can be assumed to be either 0 or 1)
    indices = np.where(mask==1)
    copied_img = cv2.copyTo(img, mask=None)
    copied_img[indices[0], indices[1], :] = [255, 255, 255]
    return copied_img


img3 = img2
imgs3 = [('Noise', img_noise_restored_simple),
         ('Gauss filter', filter_gauss(img_noise_restored_simple, 3)),
         ('Image stack + Gauss filter', filter_gauss(img_noise_stack_restored, 3))]

initial_threshold = 0.25
imgs3_gray  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in imgs3]
imgs_canny  = [filter_canny(img_gray) for img_gray in imgs3_gray]
imgs_sobel  = [filter_sobel(img_gray) for img_gray in imgs3_gray]
imgs_thresh = [applyThreshold(img_sobel, initial_threshold) for img_sobel, _, _ in imgs_sobel]
imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]

def header(label, imgs, i, j = None):
    if i == 0:
        return label, (imgs[i] if j is None else imgs[i][j])
    return imgs[i] if j is None else imgs[i][j]

imgs = [[imgs3[i], header('Sobel X'       , imgs_sobel , i, 0),
                   header('Sobel Y'       , imgs_sobel , i, 1),
                   header('Sobel'         , imgs_sobel , i, 2),
                   header('Edge mask'     , imgs_thresh, i   ),
                   header('Stylized image', imgs_masked, i   ),
                   header('Canny image'   , imgs_canny , i   )] for i in range(len(imgs3))]
imgs = [label_and_image for img_list in imgs for label_and_image in img_list]

plt.figure(figsize=(17, 7))
plt_imgs = showImages(imgs, 7, False, padding = (.05, .15, .05, .05))

def updateImg(threshold):
    imgs_thresh = [applyThreshold(img_sobel, threshold) for img_sobel, _, _ in imgs_sobel]
    imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]
    imgs_masked = [convertColorImagesBGR2RGB(img_masked)[0] for img_masked in imgs_masked]
    for i in range(len(imgs3)):
        cols = len(imgs) // len(imgs3)
        plt_imgs[i * cols + 4].set_data(imgs_thresh[i])
        plt_imgs[i * cols + 5].set_data(imgs_masked[i])

ax_threshold = plt.axes([.67, .05, .27, .06])
slider_threshold = Slider(ax=ax_threshold, label='Threshold', valmin=0, valmax=1, valinit=initial_threshold, valstep=.01)
slider_threshold.on_changed(updateImg)

plt.show()