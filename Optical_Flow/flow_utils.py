
import os
import numpy as np
import cv2



def PLACEHOLDER_FLOW(frames):
    return np.array([[[x, y] for x in np.linspace(-1, 1, frames[0].shape[1])] for y in np.linspace(-1, 1, frames[0].shape[0])])

PLACEHOLDER_FLOW_VISUALIZATION = cv2.imread('resources/example_flow_visualization.png')



#
# Task 1
#
# Implement utility functions for flow visualization.



# TODO: Convert a flow map to a BGR image for visualisation.
#       A flow map is a 2-channel 2D image with channel 1 and 2 depicting the portion flow in X and Y direction respectively.
def flowMapToBGR(flow_map):
    # TODO: Convert flow map to magnitudes and angles
    x = flow_map[:, :, 1] # cols
    y = flow_map[:, :, 0] # rows
    magnitude, angle = cv2.cartToPolar(x, y)

    # TODO: Scale and create hsv image from angles and magnitudes.
    hsv = np.zeros((flow_map.shape[0], flow_map.shape[1], 3), dtype="uint8")
    hsv[:,:,0] = angle * 180 / np.pi / 2    # H in cv2 is between 0° and 180°
    hsv[:,:,1] = 255                        # Maximum saturation.
    hsv[:,:,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert from hsv to bgr and return.
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


# TODO: Draw arrows depicting the provided `flow` on a 10x10 pixel grid.
#       You may use `cv2.arrowedLine(..)`.
def drawArrows(img, flow, arrow_color = (0, 255, 0)):
    outimg = img.copy()

    x = flow[:, :, 1] # cols
    y = flow[:, :, 0] # rows
    magnitude, angle = cv2.cartToPolar(x, y)

    # TODO: Apply proper scaling.
    scaled_flow = cv2.normalize(flow, None, -10, 10, cv2.NORM_MINMAX).astype("int32")
    #scaled_flow = flow.astype("uint8")
    #scaled_flow[:,:,0] = scaled_flow[:,:,0] * magnitude
    #scaled_flow[:,:,1] = scaled_flow[:,:,1] * magnitude

    # TODO: Draw arrows every 10th pixel.
    for row in range(0, flow.shape[0], 10):
        for col in range(0, flow.shape[1], 10):
            pt1 = (col, row)
            pt2 = (col+scaled_flow[row,col,0], row+scaled_flow[row,col,1])
            cv2.arrowedLine(outimg, pt1, pt2, arrow_color)

    return outimg


# Calculate the angular error of an estimated optical flow compared to ground truth
def calculateAngularError(estimated_flow, groundtruth_flow):
    nom = groundtruth_flow[:, :, 0] * estimated_flow[:, :, 0] + groundtruth_flow[:, :, 1] * estimated_flow[:, :, 1] + 1.0
    denom = np.sqrt((groundtruth_flow[:, :, 0] ** 2 + groundtruth_flow[:, :, 1] ** 2 + 1.0) * (estimated_flow[:, :, 0] ** 2 + estimated_flow[:, :, 1] ** 2 + 1.0))
    return (1.0 / (estimated_flow.shape[0] * estimated_flow.shape[1])) * np.sum(np.arccos(np.clip(nom / denom, 0, 1)))



# Load a flow map from a file
def load_FLO_file(filename):
    if os.path.isfile(filename) is False:
        print("file does not exist %r" % str(filename))
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    if magic != 202021.25:
        print('Magic number incorrect. .flo file is invalid')
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    # The float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # In total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    # Some cleanup (remove cv-destroying large numbers)
    flow[np.sqrt(np.sum(flow ** 2, axis = 2)) > 100] = 0
    return flow
