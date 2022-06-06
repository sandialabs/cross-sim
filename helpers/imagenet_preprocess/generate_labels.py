import os
import numpy as np
import scipy.io
from MLPerf_cal_images import get_cal_fns
from imagenet_path import imagenet_val_path

# Label array generation script from the ditsributed MAT file
path_to_val_labels = './labels/ILSVRC2012_validation_ground_truth.txt'
output_dir = "../../cross_sim/data/datasets/imagenet/"

meta = scipy.io.loadmat("./labels/meta.mat")
original_idx_to_synset = {}
synset_to_name = {}
for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
    synset = meta["synsets"][i,0][1][0]
    name = meta["synsets"][i,0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_keras_idx = {}
keras_idx_to_name = {}
f = open("./labels/synset_words.txt","r")
idx = 0
for line in f:
    parts = line.split(" ")
    synset_to_keras_idx[parts[0]] = idx
    keras_idx_to_name[idx] = " ".join(parts[1:])
    idx += 1
f.close()

f = open(path_to_val_labels,"r")
y_val = f.read().strip().split("\n")
y_val = list(map(int, y_val))
y_val = np.array([synset_to_keras_idx[original_idx_to_synset[idx]] for idx in y_val])
f.close()

np.save(output_dir+"y_val.npy", y_val)

# Generate labels for MLperf calibration set
cal_fns = get_cal_fns()
fns0 = os.listdir(imagenet_val_path)
fns0.sort()
i_cal = 0
y_val_cal = np.zeros(500)
for i in range(50000):
    if fns0[i] in cal_fns:    
        y_val_cal[i_cal] = y_val[i]
        i_cal += 1

np.save(output_dir+"y_val_MLperf_cal.npy", y_val_cal)