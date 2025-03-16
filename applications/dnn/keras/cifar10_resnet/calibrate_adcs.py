"""
Parameterizable inference simulation script for CIFAR-10 ResNets.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras
import numpy as np
import pickle
import warnings, sys, time
warnings.filterwarnings('ignore')
sys.path.append("../../") # to import dnn_inference_params
sys.path.append("../../../../") # to import simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.keras.convert import from_keras, convertible_layers, analog_layers, reinitialize
from simulator.algorithms.dnn.keras.profile import get_profiled_adc_inputs
from dnn_inference_params import dnn_inference_params
from find_adc_range import find_adc_range
from calibration import calibrate_adc_limits

# Restrict tensorflow GPU memory usage
# This seems necessary to avoid out-of-memory errors while profiling
for gpu_device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu_device, True)

## Depth parameter for model selection
# Follows definition in original ResNet paper (He et al, CVPR 2016)
# n = 2 : ResNet-14 (175K weights)
# n = 3 : ResNet-20 (272K weights)
# n = 5 : ResNet-32 (467K weights)
# n = 9 : ResNet-56 (856K weights)
n = 3

useGPU = True # use GPU?
N = 500 # number of images
batch_size = 32
Nruns = 1
print_progress = True

print("CIFAR-10: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Number of runs: {:d}".format(Nruns))
print("Batch size: {:d}".format(batch_size))

##### Load Keras model
depth = 6*n+2
model_file = "./models/resnet{:d}v1_cifar10.h5".format(depth)
resnet_model = load_model(model_file, compile=False)
n_layers = len(convertible_layers(resnet_model))

##### Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    'ideal' : False,
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
    'adc_bits' : 0,
    'adc_range_option' : "CALIBRATED",
    'adc_type' : "generic",
    'adc_per_ibit' : False,
    ## Simulation parameters
    'useGPU' : useGPU,
    ## Profiling
    'profile_xbar_inputs' : False,
    'profile_adc_inputs' : True,
    'ntest' : N
    }

### Load input limits
input_ranges = np.load("./calibrated_config/input_limits_ResNet{:d}.npy".format(depth))

### Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
    params_args_k['input_range'] = input_ranges[k]
    params_list[k] = dnn_inference_params(**params_args_k)

# Convert Keras layers to analog layers
analog_resnet = from_keras(resnet_model, params_list, fuse_batchnorm=False, bias_rows=0)
analog_resnet.compile(run_eagerly=True)

#### Load CIFAR-10 dataset
(x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Zero-center using the training set, normalize, and truncate
x_test = x_test.astype(np.float32)
x_test -= np.mean(x_train,axis=0).astype(np.float32)
x_test /= 255
x_test = x_test[:N]
y_test = y_test[:N,0]

#### Run inference and evaluate accuracy
accuracies = np.zeros(Nruns)
for m in range(Nruns):
    T1 = time.time()
    y_pred, k = np.zeros(N), 0
    outputs = analog_resnet.predict(x_test, batch_size=batch_size)
    T2 = time.time()
    y_pred = outputs.argmax(axis=1)
    top1 = np.sum(y_test == y_pred)/len(y_test)
    accuracies[m] = top1

    print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
    print('Top-1 accuracy: {:.3f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
    if m < (Nruns - 1):
        reinitialize(analog_resnet)

if Nruns > 1:
    print("==========")
    print("Mean accuracy:  {:.3f}%".format(100*np.mean(accuracies)))
    print("Stdev accuracy: {:.3f}%".format(100*np.std(accuracies)))

#### Retrieve profiled inputs and calibrate limits
print("Collecting profiled ADC data")
profiled_adc_inputs = get_profiled_adc_inputs(analog_resnet)
print("Optimizing ADC limits")
calibrated_adc_ranges = calibrate_adc_limits(
    analog_layers(analog_resnet), profiled_adc_inputs, Nbits=8)

## Make sure the file name matches the parameters used!!
np.save("./calibrated_config/adc_limits_ResNet{:d}_balanced.npy".format(depth),
    calibrated_adc_ranges)