#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#
# Task 4
#
# Load and use a pretrained `Object Detection` model from the `TensorFlow Model Garden`

# Download/cache required model and test data

from tf_utils import *

print('Download model...')
modelPath = download_model('20200713', 'centernet_hg104_1024x1024_coco17_tpu-32')
print(modelPath)

print('Download labels...')
labelsPath = download_labels('mscoco_label_map.pbtxt')
print(labelsPath)

print('Download test images...')
imagePaths = download_test_images(['image1.jpg', 'image2.jpg'])
print('\n'.join(imagePaths))



# Load the model

import tensorflow as tf
import os
from utils import *

print('Load model')

# TODO: Load the downloaded saved tensorflow model using `load_model(..)` from the keras module
savedModelPath = os.path.join(modelPath, "saved_model")
model = tf.keras.models.load_model(filepath=savedModelPath)



# Load label map data (for plotting)

from object_detection.utils import label_map_util

print('Load labels')

category_index = label_map_util.create_category_index_from_labelmap(labelsPath, use_display_name=True)



# Run inference

import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') # Reactivate GUI backend (deactivated by `import viz_utils`)

imgs = []
for image_path in imagePaths:
    print('Running inference for {}... '.format(image_path))
    img = cv2.imread(image_path)
    # TODO: Convert image to a tensor.
    img_tensor = tf.convert_to_tensor(img)
    # TODO: Add a new batch dimension.
    img_tensor = tf.expand_dims(img_tensor, -1)

    # TODO: Detect the objects in `img` using the loaded model
    #res = model.predict(img_tensor)

    # TODO: Add bounding boxes and labels of the detected objects to `img` using `viz_utils.visualize_boxes_and_labels_on_image_array(..)`.
    #  Tipps: All required data is either already available or contained in the dict-like structure returned when applying the model to `img`.
    #         TensorFlow models return tensors with an additional batch dimension that is not required for visualitation, i.e. take only index[0] and convert it to a numpy array.
    #         TensorFlow returns detected classes as floats, but the visualization requires ints.
    #
    # ???
    #

    #imgs.append(img)
plt.figure(figsize=(13, 5))
showImages(imgs)
