#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import os, sys, pickle
#To import parameters
sys.path.append("../../../simulator/")
#To import simulator
sys.path.append("../../../")
#To import dataset loader
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from interface.inference_net import set_params, inference
from interface.keras_parser import get_keras_metadata
from interface.dnn_setup import augment_parameters, build_keras_model, model_specific_parameters, \
    get_xy_parallel, get_xy_parallel_parasitics, load_adc_activation_ranges
from interface.config_message import print_configuration_message

# ==========================
# ==== Load config file ====
# ==========================

import inference_config as config

# ===================
# ==== GPU setup ====
# ===================

# Restrict tensorflow GPU memory usage
os.environ["CUDA_VISIBLE_DEVICES"]=str(-1)
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for k in range(len(gpu_devices)):
    tf.config.experimental.set_memory_growth(gpu_devices[k], True)

# Set up GPU
if config.useGPU:
    import cupy
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    cupy.cuda.Device(0).use()

# =====================================
# ==== Import neural network model ====
# =====================================

# Build Keras model and prepare CrossSim-compatible topology metadata
keras_model = build_keras_model(config.model_name,show_model_summary=config.show_model_summary)
layerParams, sizes = get_keras_metadata(keras_model,task=config.task,debug_graph=False)

# Count the total number of layers and number of MVM layers
Nlayers = len(layerParams)
config.Nlayers_mvm = np.sum([(layerParams[j]['type'] in ('conv','dense')) for j in range(Nlayers)])

# ===================================================================
# ======= Parameter validation and model-specific parameters ========
# ===================================================================
# General parameter check
config = augment_parameters(config)
# Parameters specific to some neural networks
config, positiveInputsOnly = model_specific_parameters(config)

# =========================================================
# ==== Load calibrated ranges for ADCs and activations ====
# =========================================================

adc_ranges, dac_ranges = load_adc_activation_ranges(config)

# =======================================
# ======= GPU performance tuning ========
# =======================================

# Convolutions: number of sliding windows along x and y to compute in parallel
xy_pars = get_xy_parallel(config, disable=config.disable_SW_packing)

# ================================
# ========= Start sweep ==========
# ================================

# Display the chosen simulation settings
print_configuration_message(config)

for q in range(config.Nruns):

    if config.Nruns > 1:
        print('')
        print('===========')
        print(" Run "+str(q+1)+"/"+str(config.Nruns))
        print('===========')

    paramsList, layerParamsCopy = Nlayers*[None], Nlayers*[None]
    j_mvm, j_conv = 0, 0 # counter for MVM and conv layers

    # ===================================================
    # ==== Compute and set layer-specific parameters ====
    # ===================================================

    for j in range(Nlayers):

        # For a layer that must be split across multiple arrays, create multiple params objects
        if layerParams[j]['type'] in ('conv','dense'):

            # Number of total rows used in MVM
            if layerParams[j]['type'] == 'conv':
                Nrows = layerParams[j]['Kx']*layerParams[j]['Ky']*layerParams[j]['Nic']
                Ncols = layerParams[j]['Noc']
                if not config.digital_bias:
                    if layerParams[j]['bias'] or config.fold_batchnorm:
                        Nrows += 1

            elif layerParams[j]['type'] == 'dense':
                Nrows = sizes[j][2]
                Ncols = sizes[j+1][2]
                if not config.digital_bias and layerParams[j]['bias']:
                    Nrows += 1

            # Compute number of arrays matrix must be partitioned across
            if config.NrowsMax > 0:
                Ncores = (Nrows-1)//config.NrowsMax + 1
            else:
                Ncores = 1

            if config.NcolsMax > 0:
                Ncores *= (Ncols-1)//config.NcolsMax + 1

            # Layer specific ADC and activation resolution and range (set above)
            adc_range = adc_ranges[j_mvm]
            if adc_range is not None:
                adc_range = adc_range.tolist()
            dac_range = dac_ranges[j_mvm]
            adc_bits_j = config.adc_bits_vec[j_mvm]
            dac_bits_j = config.dac_bits_vec[j_mvm]
            if config.gate_input:
                Rp_j = config.Rp_col
            else:
                Rp_j = np.maximum(config.Rp_row,config.Rp_col)

            # If parasitics are enabled, x_par and y_par are modified to optimize cumulative sum runtime
            if Rp_j > 0 and layerParams[j]['type'] == 'conv':
                xy_pars[j_conv,:] = get_xy_parallel_parasitics(Nrows,sizes[j][0],sizes[j+1][0],config.model_name,
                    disable=config.disable_SW_packing)

            if layerParams[j]['type'] == 'conv':
                x_par, y_par = xy_pars[j_conv,:]
                convParams = layerParams[j]

            elif layerParams[j]['type'] == 'dense':
                x_par, y_par = 1, 1
                convParams = None

            if config.Icol_max > 0:
                Icol_max_norm = config.Icol_max / config.Icell_max
            else:
                Icol_max_norm = 0

            # Does this layer use analog batchnorm?
            analog_batchnorm = config.fold_batchnorm and layerParams[j]['batch_norm'] is not None

            params = set_params(task=config.task,
                wtmodel=config.style,
                convParams=convParams,
                alpha_noise=config.alpha_noise,
                balanced_style=config.balanced_style,
                ADC_per_ibit=config.ADC_per_ibit,
                x_par=x_par,
                y_par=y_par,
                weight_bits=config.weight_bits,
                weight_percentile=config.weight_percentile,
                useGPU=config.useGPU,
                proportional_noise=config.proportional_noise,
                alpha_error=config.alpha_error,
                adc_bits=adc_bits_j,
                dac_bits=dac_bits_j,
                adc_range=adc_range,
                dac_range=dac_range,
                error_model=config.error_model,
                noise_model=config.noise_model,
                NrowsMax=config.NrowsMax,
                NcolsMax=config.NcolsMax,
                positiveInputsOnly=positiveInputsOnly[j_mvm],
                input_bitslicing=config.input_bitslicing,
                gate_input=config.gate_input,
                subtract_current_in_xbar=config.subtract_current_in_xbar,
                interleaved_posneg=config.interleaved_posneg,
                t_drift=config.t_drift,
                drift_model=config.drift_model,
                Rp_row=config.Rp_row,
                Rp_col=config.Rp_col,
                digital_offset=config.digital_offset,
                Icol_max=Icol_max_norm,
                infinite_on_off_ratio=config.infinite_on_off_ratio,
                Rmin = config.Rmin,
                Rmax = config.Rmax,
                adc_range_option=config.adc_range_option,
                proportional_error=config.proportional_error,
                Nslices=config.Nslices,
                digital_bias=config.digital_bias,
                analog_batchnorm=analog_batchnorm,
                adc_type=config.adc_type,
                input_slice_size=config.input_slice_size)

            if Ncores == 1:
                paramsList[j] = params
            else:
                paramsList[j] = Ncores*[None]
                for k in range(Ncores):
                    paramsList[j][k] = params.copy()            
            
            j_mvm += 1
            if layerParams[j]['type'] == 'conv':
                j_conv += 1

         # Need to make a copy to prevent inference() from modifying layerParams
        layerParamsCopy[j] = layerParams[j].copy()

    # Run inference
    accuracy, network_output = inference(ntest=config.ntest,
        dataset=config.task,
        paramsList=paramsList,
        sizes=sizes,
        keras_model=keras_model,
        layerParams=layerParamsCopy,
        useGPU=config.useGPU,
        count_interval=config.count_interval,
        randomSampling=config.randomSampling,
        topk=config.topk,
        subtract_pixel_mean=config.subtract_pixel_mean,
        memory_window=config.memory_window,
        model_name=config.model_name,
        fold_batchnorm=config.fold_batchnorm,
        digital_bias=config.digital_bias,
        nstart=config.nstart,
        ntest_batch=config.ntest_batch,
        bias_bits=config.bias_bits,
        time_interval=config.time_interval,
        imagenet_preprocess=config.imagenet_preprocess,
        dataset_normalization=config.dataset_normalization,
        adc_range_option=config.adc_range_option,
        show_HW_config=config.show_HW_config,
        return_network_output=config.return_network_output)

    # Collect network outputs
    if config.return_network_output:
        if q == 0:
            network_outputs = np.zeros((config.Nruns, config.ntest, network_output.shape[1]))
        network_outputs[q,:,:] = network_output

# Save the network outputs
# Outputs will be saved to a .npy file, as a 3D array
# The 3D array has dimensions: # runs (Nruns) x # input examples (ntest) x # network outputs
if config.return_network_output:
    output_directory = './network_outputs/'
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    # A more descriptive file name is recommended
    filename = output_directory+config.model_name+"_Nruns="+str(config.Nruns)+"_outputs.npy"
    np.save(filename, network_outputs)
    print("Saved network outputs to: "+filename)
