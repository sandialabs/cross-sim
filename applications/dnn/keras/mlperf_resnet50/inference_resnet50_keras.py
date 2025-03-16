"""
Parameterizable inference simulation script for MLPerf's ResNet50-v1.5
benchmark model on ImageNet.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
import tensorflow as tf
import keras
import numpy as np
import pickle
import warnings, sys, time
from buildResnet50v15 import ResNet50v15
warnings.filterwarnings('ignore')
sys.path.append("../../") # to import dnn_inference_params
sys.path.append("../../../../") # to import simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.keras.convert import from_keras, convertible_layers, reinitialize
from dnn_inference_params import dnn_inference_params
from find_adc_range import find_adc_range

useGPU = True # use GPU?
N = 1000 # number of images
batch_size = 8
Nruns = 1
print_progress = True

print("ImageNet: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Number of runs: {:d}".format(Nruns))
print("Batch size: {:d}".format(batch_size))

##### Load Keras model
weights_file = "resnet50v15-weights.p"
weights_dict = pickle.load(open(weights_file,"rb"))
resnet50_model = ResNet50v15(weights_dict)
n_layers = len(convertible_layers(resnet50_model))

##### Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    'ideal' : True,
    ## Mapping style
    'core_style' : "BALANCED",
    'Nslices' : 1,
    ## Weight value representation and precision
    'weight_bits' : 8,
    'weight_percentile' : 100,
    'digital_bias' : True,
    ## Memory device
    'Rmin' : 1e4,
    'Rmax' : 1e6,
    'infinite_on_off_ratio' : False,
    'error_model' : "none",
    'alpha_error' : 0.00,
    'proportional_error' : False,
    'noise_model' : "none",
    'alpha_noise' : 0.00,
    'proportional_noise' : False,
    'drift_model' : "none",
    't_drift' : 0,
    ## Array properties
    'NrowsMax' : 1152, # inputs
    'NcolsMax' : 256, # outputs
    'Rp_row' : 0, # ohms
    'Rp_col' : 0, # ohms
    'interleaved_posneg' : False,
    'subtract_current_in_xbar' : True,
    'current_from_input' : True,
    ## Input quantization
    'input_bits' : 8,
    'input_bitslicing' : False,
    'input_slice_size' : 1,
    ## ADC
    'adc_bits' : 8,
    'adc_range_option' : "CALIBRATED",
    'adc_type' : "generic",
    'adc_per_ibit' : False,
    ## Simulation parameters
    'useGPU' : useGPU
    }

### Load input limits
input_ranges = np.load("./calibrated_config/dac_limits_ResNet50v15.npy")

### Load ADC limits
adc_ranges = find_adc_range(base_params_args, n_layers)

### Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
    params_args_k['input_range'] = input_ranges[k]
    params_args_k['adc_range'] = adc_ranges[k]
    params_list[k] = dnn_inference_params(**params_args_k)

# Convert Keras layers to analog layers
analog_resnet50 = from_keras(resnet50_model, params_list, fuse_batchnorm=True, bias_rows=0)
analog_resnet50.compile(run_eagerly=True)

#### Load pre-processed ImageNet dataset
imagenet_path = "../../../../../../../imagenet/"
if N <= 1000:
    x = np.load(imagenet_path + "x_val_MLperfRN50_1000.npy")[:N,:,:,:]
elif N <= 25000:
    x = np.load(imagenet_path + "x_val_MLperfRN50_25000_front.npy")[:N,:,:,:]
y = np.load(imagenet_path + "y_val.npy")[:N] + 1

#### Run inference and evaluate accuracy
accuracies = np.zeros(Nruns)
for m in range(Nruns):
    T1 = time.time()
    y_pred, k = np.zeros(N), 0
    outputs = analog_resnet50.predict(x, batch_size=batch_size)
    T2 = time.time()

    y_pred = outputs.argmax(axis=1)
    top1 = np.sum(y == y_pred)/len(y)
    accuracies[m] = top1

    print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
    print('Top-1 accuracy: {:.3f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
    if m < (Nruns - 1):
        reinitialize(analog_resnet50)

if Nruns > 1:
    print("==========")
    print("Mean accuracy:  {:.3f}%".format(100*np.mean(accuracies)))
    print("Stdev accuracy: {:.3f}%".format(100*np.std(accuracies)))