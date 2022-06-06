import os
import numpy as np
from PIL import Image
from torchvision import transforms
from MLPerf_cal_images import get_cal_fns
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

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

for i in range(n_start,n_end):

    if not MLperf_calibration or (MLperf_calibration and fns0[i] in cal_fns):
        if not MLperf_calibration and i %100 == 0:
            print("%d/%d" % (i, n_end))
        elif MLperf_calibration and i_cal % 10 == 0:
            print("Calibration image "+str(i_cal+1)+"/500")

        img = Image.open(fns[i])
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = preprocess(img)
        img = img.numpy()
        img = np.clip(np.round(img*127.0/2.64), -127.0, 127.0)
        img = np.asarray(img, dtype='float32')
        img = img.transpose(1,2,0)

        # Save (as RGB)
        if not MLperf_calibration:
            x_val[i-n_start,:,:,:] = img
        else:
            x_val[i_cal,:,:,:] = img
            i_cal += 1

if not MLperf_calibration:
    if num_images == 25000:
        if n_start == 0:
            np.save(output_dir+"./x_val_torch_"+str(num_images)+"_front.npy", x_val)
        elif n_start == 25000:
            np.save(output_dir+"./x_val_torch_"+str(num_images)+"_back.npy", x_val)
    else:
        np.save(output_dir+"./x_val_torch_"+str(num_images)+".npy", x_val)
else:
    np.save(output_dir+"./x_val_torch_cal.npy", x_val)
