import os
import numpy as np
import cv2
from imagenet_path import imagenet_val_path

# Path to save preprocessed .npy arrays to
output_dir = "../../cross_sim/data/datasets/imagenet/"

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

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

means = np.array([123.68, 116.78, 103.94], dtype=np.float32)

for i in range(n_start,n_end):

    if not MLperf_calibration or (MLperf_calibration and fns0[i] in cal_fns):
        if not MLperf_calibration and i %100 == 0:
            print("%d/%d" % (i, n_end))
        elif MLperf_calibration and i_cal % 10 == 0:
            print("Calibration image "+str(i_cal+1)+"/500")
        
        # Load (as BGR)
        img = cv2.imread(fns[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        output_height = 224
        output_width = 224
        cv2_interpol = cv2.INTER_AREA
        img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
        img = center_crop(img, output_height, output_width)
        img = np.asarray(img, dtype='float32')    
        img -= means

        # Save (as RGB)
        if not MLperf_calibration:
            x_val[i-n_start,:,:,:] = img
        else:
            x_val[i_cal,:,:,:] = img
            i_cal += 1

if not MLperf_calibration:
    if num_images == 25000:
        if n_start == 0:
            np.save(output_dir+"x_val_MLperfRN50_"+str(num_images)+"_front.npy", x_val)
        elif n_start == 25000:
            np.save(output_dir+"x_val_MLperfRN50_"+str(num_images)+"_back.npy", x_val)
    else:
        np.save(output_dir+"x_val_MLperfRN50_"+str(num_images)+".npy", x_val)
else:
    np.save(output_dir+"x_val_MLperfRN50_cal.npy", x_val)