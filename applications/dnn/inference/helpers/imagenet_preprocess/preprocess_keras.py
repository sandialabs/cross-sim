#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import os
import numpy as np
import cv2
from MLPerf_cal_images import get_cal_fns
from imagenet_path import imagenet_val_path

# Path to save preprocessed .npy arrays to
output_dir = "../../../data/datasets/imagenet/"

# Whether nor not preprocess images in the MLPerf calibration set
MLperf_calibration = False

# If not using calibration set, how many images to preprocess and the index to start from
n_start = 0
num_images = 50

if not MLperf_calibration:
    n_end = n_start + num_images
else:
    num_images = 500
    n_start, n_end = 0, 50000
    cal_fns = get_cal_fns()
    i_cal = 0

# Initialize data matrix
x_val = np.zeros((num_images, 224, 224, 3), dtype=np.float32)

# Get file names
fns0 = os.listdir(imagenet_val_path)
fns0.sort()
fns = [imagenet_val_path + fn for fn in fns0]

# Preprocess using OpenCV
for i in range(n_start,n_end):

    if not MLperf_calibration or (MLperf_calibration and fns0[i] in cal_fns):
        if not MLperf_calibration and i %100 == 0:
            print("%d/%d" % (i, n_end))
        elif MLperf_calibration and i_cal % 10 == 0:
            print("Calibration image "+str(i_cal+1)+"/500")
        
        ## Load (as BGR)
        img = cv2.imread(fns[i])
        
        # Resize
        height, width, _ = img.shape
        new_height = height * 256 // min(img.shape[:2])
        new_width = width * 256 // min(img.shape[:2])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Crop
        height, width, _ = img.shape
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        img = img[starty:starty+224,startx:startx+224]
        assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
        
        # Save (as RGB)
        if not MLperf_calibration:
            x_val[i-n_start,:,:,:] = img[:,:,::-1]
        else:
            x_val[i_cal,:,:,:] = img[:,:,::-1]
            i_cal += 1


if not MLperf_calibration:
    if num_images == 25000:
        if n_start == 0:
            np.save(output_dir+"x_val_keras_"+str(num_images)+"_front.npy", x_val)
        elif n_start == 25000:
            np.save(output_dir+"x_val_keras_"+str(num_images)+"_back.npy", x_val)
    else:
        np.save(output_dir+"x_val_keras_"+str(num_images)+".npy", x_val)
else:
    np.save(output_dir+"x_val_keras_cal.npy", x_val)