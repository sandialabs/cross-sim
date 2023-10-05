#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# ==============================================
# ========== Machine settings ==================
# ==============================================

# Enable GPU compute?
useGPU = False

# Which GPU to use (set to 0 if only one GPU)
gpu_num = 0

# Number of runs with identical settings, but different random seeds (if applicable)
Nruns = 1

# ==============================================
# ======= Dataset and model settings ===========
# ==============================================

# task = "imagenet"
# task = "cifar100"
# task = "cifar10"
task = "mnist"

# Choose neural network model based on task
if task == "imagenet":
    # model_name = "Resnet50"
    model_name = "Resnet50-v1.5"
    # model_name = "Resnet50-int4"
    # model_name = "VGG19" ## weights will be downloaded from keras.applications
    # model_name = "InceptionV3"
    # model_name = "MobilenetV2"
    # model_name = "MobilenetV1"
    # model_name = "MobilenetV1-int8"

elif task == "cifar100":
    model_name = "ResNet56_cifar100"

elif task == "cifar10":
    # model_name = "cifar10_cnn_brelu"
    model_name = "ResNet14"
    # model_name = "ResNet20"
    # model_name = "ResNet32"
    # model_name = "ResNet56"

elif task == "mnist":
    # model_name = 'lenet5'
    # model_name = "CNN6"
    model_name = "CNN6_v2"

# Dataset truncation
ntest = 1000 # number of images in inference simlation
ntest_batch = 1000 # how many images to load at a time in one contiguous block (for ImageNet, should be <=5000)
nstart = 0 # index of starting image

# Random sampling: 
# If True, ntest images will be chosen randomly from the full dataset
# If False, images will be loaded in the order that they are stored
randomSampling = False

# Console outputs
# top-k accuracy to record
# count_interval: cumulative accuracy printed after every N images (N = count_interval)
# time_interval: print the time elapsed between the processing of N images
if task == "imagenet":
    count_interval = 1
    topk = (1,5)
elif task == "cifar10" or task == "cifar100":
    count_interval = 10
    topk = 1
elif task == "mnist":
    count_interval = 100
    topk = 1
time_interval = True

# Return the output of the final layer (before argmax) in addition to classification accuracy
# This is useful if performing regression
return_network_output = False

# Show the Keras model summary
show_model_summary = False

# Show analog hardware configuration
show_HW_config = False

# Disable sliding window packing?
# SW packing is only used to speed up simulations if read noise or parasitic resistance is enabled
# May need to tune SW packing parameters to optimize speedup (see dnn_setup.get_xy_parallel)
disable_SW_packing = False

# ==============================================
# ========= Crossbar configuration =============
# ==============================================

# Resolution of the weight values
weight_bits = 8

# Percentile of weight distribution in each layer that will correspond to maximum device conductance
# In general, weight_percentile = 100 is highly recommended
weight_percentile = 100

# Number of bit slices
Nslices = 1

# Max number of rows
NrowsMax = 1152

# Max number of columns
NcolsMax = 0

# Negative number handling (BALANCED or OFFSET)
style = "BALANCED"
# style = "OFFSET"

# Special option for "BALANCED" *and* Nslices = 1, otherwise ignored
#     ONE_SIDED: zero value maps both devices to lowest state (recommended)
#     TWO_SIDED: zero value maps both devices to center state
# balanced_style = "TWO_SIDED"
balanced_style = "ONE_SIDED"

# Special option for "BALANCED" *and* interleaved_posneg = False, otherwise ignored
# Whether to subtract currents from positive and negative crossbars/columns in analog.
# If False, both analog outputs are digitized separately then subtracted digitally
subtract_current_in_xbar = True

# Special option for "OFFSET", otherwise ignored
# Whether offset is computed digitally (True) or using an analog zero-point column (False) 
digital_offset = True

# Array unit cell parasitic resistance
Rp_row = 0 # ohms
Rp_col = 0 # ohms

# Whether parasitic voltage drops along a row should be set to zero
gate_input = False

# Interleave positive and negative cells in one column; ignore if OFFSET
interleaved_posneg = False

# Clip the current at the bottom of a column to the range (-Icol_max,+Icol_max)
# Applied on each input bit if input_bitslicing is True
# Applied prior to current subtraction if BALANCED, unless interleaved
Icol_max = 0
Icell_max = 1.8e-6 # Max possible cell current; used only if Icol_max > 0

# Fold batchnorm into conv/dense layer matrix
# If False, batchnorm will be computed digitally after the MVM
fold_batchnorm = True

# Implement bias digitally vs in array
# If False, will likely require higher weight, DAC, and ADC precision
digital_bias = True

# Bias weight resolution
#   0:      no quantization
#   1+:     uniform bias bit resolution
#   adc:    variable bit resolution, track ADC level spacing if ADC is on
bias_bits = 0
# bias_bits = "adc"

# ==============================================
# ========= Weight non-idealities ==============
# ==============================================

# Cell Resistance, infinite_on_off_ratio ignores Rmax because it is infinite
Rmin = 1e3 # ohms
Rmax = 1e6 # ohms
infinite_on_off_ratio = True

###############
#
#   To use a custom rather than generic device error model, please implement
#   a method in one of the files in the directory:
#   /cross_sim/cross_sim/xbar_simulator/parameters/custom_device/
#   -- Programming errors:         weight_error_device_custom.py
#   -- Cycle-to-cycle read noise:  weight_readnoise_device_custom.py
#   -- Conductance drift:          weight_drift_device_custom.py
#   For more details, see Chapter 7 of the Inference manual.

### Programming error
# error_model can be (str): "none", "generic" (generic), or custom device model
#   Custom devices are defined in individual files in /simulator/devices/custom/
#   Examples: "SONOS", "PCMJoshi", "RRAMMilo"
error_model = "none"
alpha_error = 0.00 # used only if error_model is generic
proportional_error = True # used only if error_model is generic

### Read noise
# noise_model can be (str): "none", "generic" (generic), or custom device model
#   Custom devices are defined in individual files in /simulator/devices/custom/
#   Examples: "SONOS"
noise_model = "none"
alpha_noise = 0.00 # used only if noise_model is generic
proportional_noise = True # used only if noise_model is generic

### Conductance drift
# drift_model can be (str): "none", or custom device model
#   Custom devices are defined in individual files in /simulator/devices/custom/
#   Examples: "SONOS"
# NOTE: drift model may apply programming error if t_drift = 0
# To fully disable drift model, set drift_model = "none"
t_drift = 0 # time after programming (days)
drift_model = "none"

# ==============================================
# ===== ADC and activation (DAC) settings ======
# ==============================================

# Resolution: 0 means no quantization
adc_bits = 0
dac_bits = 8

# Simulate input bitslicing
input_bitslicing = False

# If input bit slicing enabled, # bits per input slice
# This corresponds to the resolution of the input DAC
input_slice_size = 1

# Digitization after every input bit (ignored if input_bit_slicing is False)
# Recommended settings: False if BALANCED, True if OFFSET
ADC_per_ibit = False

# ADC range option
#   CALIBRATED: use saved ADC ranges for every layer
#       For non-bitsliced, this gives the minimum and maximum ADC level, whether ADC is applied after each input bit or not
#       For bit sliced, this gives the values of N_i (integer), where the extreme ADC level is MAX / 2^N_i for the i^th slice
#   MAX: set ADC ranges to maximum possible for each layer
#   GRANULAR: set ADC range according to # bits and a fixed level separation corresponding to the FPG (ADC_per_ibit must be True)
#
adc_range_option = "CALIBRATED"
# adc_range_option = "MAX"
# adc_range_option = "GRANULAR"

# ADC implementation
# Currently supported: "generic", "ramp", "sar", "pipeline", "cyclic"
# Ignored if adc_bits = 0
adc_type = "generic"
