#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np

from flow_utils import *
from utils import *


#
# Task 3
#
# Load and use a pretrained model to estimate the optical flow of the same two frames as in Task 2.


# Load image frames
frames = [  cv2.imread("resources/frame1.png"),
            cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")



# TODO: Load the model.
#
# ???
#



# TODO: Run model inference on the two frames.
#
# ???
#



# Create and show visualizations for the computed flow
#
# ???
#



