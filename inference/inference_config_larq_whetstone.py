#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
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

task = "cifar10"

model_name = "larq_cifar10"
# model_name = "whetstone_cifar10"

# Dataset truncation
ntest = 1000 # number of images in inference simlation
ntest_batch = ntest # how many images to load at a time in one contiguous block (for ImageNet, should be <=5000)
nstart = 0 # index of starting image

# Random sampling: 
# If True, ensure ntest_batch > ntest. ntest images will be chosen randomly from the batch.
# If False, images will be loaded in the order that they are stored
randomSampling = False

# Console outputs
# top-k accuracy to record
# count_interval: cumulative accuracy printed after every N images (N = count_interval)
# time_interval: print the time elapsed between the processing of N images
count_interval = 10
topk = 1
time_interval = True

# Show the Keras model summary
show_model_summary = False

# ==============================================
# ========= Crossbar configuration =============
# ==============================================

# Resolution of the weight values
weight_bits = 0

# Percentile of weight distribution in each layer that will correspond to maximum device conductance
weight_percentile = 100

# Number of bit slices
Nslices = 1

# Max number of rows
NrowsMax = 0

# Negative number handling (BALANCED or OFFSET)
style = "BALANCED"

# Special option for "BALANCED", otherwise ignored
# one_sided: zero value maps both devices to lowest state (recommended)
# two_sided: zero value maps both devices to center state
balanced_style = "one_sided"

# Special option for "OFFSET", otherwise ignored
# Whether offset is computed digitally (True) or using an analog zero-point column (False) 
digital_offset = True

# Simulate input bitslicing
input_bitslicing = False

# Array unit cell parasitic resistance
Rp = 0

# Whether parasitic voltage drops along a row should be set to zero
noRowParasitics = True

# Interleave positive and negative cells in one column; ignore if OFFSET
interleaved_posneg = False

# Clip the current at the bottom of a column to the range (-Icol_max,+Icol_max)
# Applied on each input bit if input_bitslicing is True
# Applied prior to current subtraction if BALANCED, unless interleaved
Icol_max = 0
Icell_max = 3.2e-6 # Max possible cell current; used only if Icol_max > 0

# Batchnorm and bias weights
if model_name == "larq_cifar10":
    fold_batchnorm = False
elif model_name == "whetstone_cifar10":
    fold_batchnorm = True
digital_bias = True

# Bias weight resolution
# 0: no quantization
# adc: track ADC if ADC is on, no quantization otherwise
bias_bits = 0

# ==============================================
# ========= Weight non-idealities ==============
# ==============================================

# Cell condcutance On/off ratio: 0 means infinity
On_off_ratio = 0

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
# error_model can be (str): "none", "alpha" (generic), or custom device model
#   Available device models are: "SONOS", "PCM_Joshi", "RRAM_Milo"
#   Define in weight_error_device_custom.py
error_model = "alpha"
alpha_error = 0.00 # used only if error_model is alpha
proportional_error = False # used only if error_model is alpha

### Read noise
# noise_model can be (str): "none", "alpha" (generic), or custom device model
#   Available device models are: "parabolic" (hypothetical) 
#   Define in weight_readnoise_device_custom.py
noise_model = "alpha"
alpha_noise = 0.00 # used only if noise_model is alpha
proportional_noise = False # used only if noise_model is alpha

### Conductance drift
# Drift model is disabled if t_drift = 0
# drift_model can be (str): "none", or custom device model
#   Available device models are: "SONOS_interpolate", "PCM_Joshi"
#   Define in weight_drift_device_custom.py
t_drift = 0 # time after programming (days)
drift_model = 'none'

# ==============================================
# ===== ADC and activation (DAC) settings ======
# ==============================================

# Resolution: 0 means no quantization
adc_bits = 0
dac_bits = 0

# Digitization after every input bit (ignored if input_bit_slicing is False)
# Recommended settings: BALANCED -> ADC_per_ibit = False
#                       OFFSET ->   ADC_per_ibit = True
ADC_per_ibit = (False if style == "BALANCED" else True)

# ADC range option
#   calibrated: use saved ADC ranges for every layer
#       For non-bitsliced, this gives the minimum and maximum ADC level, whether ADC is applied after each input bit or not
#       For bit sliced, this gives the values of N_i (integer), where the extreme ADC level is MAX / 2^N_i for the i^th slice
#   max: set ADC ranges to maximum possible for each layer
#   FPG: set ADC range according to # bits and a fixed level separation corresponding to the FPG (ADC_per_ibit must be True)
adc_range_option = "calibrated"

# Percentile option to select a calibrated range, if using bit slicing (otherwise ignored)
# Typical values are 99.9, 99.95, 99.99, 99.995, and 99.999
pct = 99.99