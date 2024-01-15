#!/usr/bin/env python3

import numpy as np
import cv2

def filter_box(img, ksize=5):
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

    """
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
    """
    return output


# Load image.
img = cv2.imread("img/geometric_shapes.png")
cv2.imshow("img", img)

# Box filter.
# box_filtered_img = filter_box(img, 11)
# cv2.imshow("box_filtered_img", box_filtered_img)

# Sin filter.
sinc_filtered_img = filter_sinc(img, 0.4)
cv2.imshow("sinc_filtered_img", sinc_filtered_img)

cv2.waitKey(0)
