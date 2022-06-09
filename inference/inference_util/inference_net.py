#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys, io, os, time, random
import numpy as np
import tensorflow.keras.backend as K
sys.path.append("..")
from cross_sim import Backprop, Parameters
from helpers.dataset_loaders import load_data_mnist, load_data_fashion_mnist, load_cifar_10, load_cifar_100, load_imagenet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_params(**kwargs):
    """
    Pass parameters using kwargs to allow for a general parameter dict to be used
    This function should be called before train and sets all parameters of the neural core simulator

    :return: params, a parameter object with all the settings

    """
    #######################
    #### load relevant parameters from arg dict
    task = kwargs.get("task","small")
    wtmodel = kwargs.get("wtmodel","BALANCED")
    convParams = kwargs.get("convParams",None)

    error_model = kwargs.get("error_model","none")
    alpha_error = kwargs.get("alpha_error",0.0)
    proportional_error = kwargs.get("proportional_error",False)

    noise_model = kwargs.get("noise_model","none")
    alpha_noise = kwargs.get("alpha_noise",0.0)
    proportional_noise = kwargs.get("proportional_noise",False)

    t_drift = kwargs.get("t_drift",0)
    drift_model = kwargs.get("drift_model",None)

    Rp = kwargs.get("Rp",0)

    NrowsMax = kwargs.get("NrowsMax",None)
    weight_bits = kwargs.get("weight_bits",0)
    adc_bits = kwargs.get("adc_bits",8)
    dac_bits = kwargs.get("dac_bits",8)
    dac_range = kwargs.get("dac_range",(0,1))
    adc_range = kwargs.get("adc_range",(0,1))
    positiveInputsOnly = kwargs.get("positiveInputsOnly",False)
    interleaved_posneg = kwargs.get("interleaved_posneg",False)
    On_off_ratio = kwargs.get("On_off_ratio",0) # zero means infinity
    noRowParasitics = kwargs.get("noRowParasitics",True)

    Nslices = kwargs.get("Nslices",1)    
    digital_offset = kwargs.get("digital_offset",False)
    adc_range_option = kwargs.get("adc_range_option",False)
    Icol_max = kwargs.get("Icol_max",1e6)
    digital_bias = kwargs.get("digital_bias",False)

    x_par = kwargs.get("x_par",1)
    y_par = kwargs.get("y_par",1)
    weight_reorder = kwargs.get("weight_reorder",True)
    useGPU = kwargs.get("useGPU",False)
    gpu_id = kwargs.get("gpu_id",0)
    profile_ADC_inputs = kwargs.get("profile_ADC_inputs",False)

    balanced_style = kwargs.get("balanced_style","one_sided")
    input_bitslicing = kwargs.get("input_bitslicing",False)
    fast_balanced = kwargs.get("fast_balanced",False)
    ADC_per_ibit = kwargs.get("ADC_per_ibit",False)
    noRpSlices = kwargs.get("noRpSlices",None)

    ################  create parameter objects with all neural core settings for first core
    params = Parameters()

    #set core type
    if Nslices == 1:
        params.algorithm_params.crossbar_type = wtmodel
    else:
        params.algorithm_params.crossbar_type = "BITSLICED"

    params.algorithm_params.sim_type="NUMERIC"

    # GPU acceleration?
    params.numeric_params.useGPU = useGPU
    if useGPU:
        params.numeric_params.gpu_id = gpu_id

    # Multiple convolutional VMMs in parallel? (intended for GPU, but possible to use w/ CPU as well)
    # Note: GPU acceleration possible for both FC and conv layers, but x_par and y_par only valid for conv layers
    params.numeric_params.x_par = x_par # Number of sliding window steps to do in parallel (x)
    params.numeric_params.y_par = y_par # Number of sliding window steps to do in parallel (y)

    # Whether to reorder the duplicated weights into Toeplitz form when x_par, y_par > 1 so that the
    # expanded matrix takes up less memory
    # This option can only be implemented when the following conditions are met
    if x_par == 1 and y_par == 1:
        weight_reorder = False
    if convParams is None:
        weight_reorder = False
    else:
        # These cases have no input reuse to exploit
        if convParams['Kx'] == 1 and convParams['Ky'] == 1:
            weight_reorder = False
        if (x_par > 1 and convParams['stride'] >= convParams['Kx']) and \
            (y_par > 1 and convParams['stride'] >= convParams['Ky']):
            weight_reorder = False
        # Cases that don't make sense to implement (as of now)
        if NrowsMax < convParams['Kx']*convParams['Ky']*convParams['Nic']:
            weight_reorder = False
        if convParams['bias'] and not digital_bias:
            weight_reorder = False
        if wtmodel == "OFFSET":
            weight_reorder = False
        if Rp > 0:
            weight_reorder = False
        if noise_model == "alpha" and alpha_noise > 0:
            weight_reorder = False
        if noise_model != "none" and noise_model != "alpha":
            weight_reorder = False

    params.numeric_params.weight_reorder = weight_reorder
    
    # set convolutional layer parameters
    if convParams is not None:
        params.convolution_parameters.is_conv_core = (convParams['type'] == 'conv')
        params.convolution_parameters.sameConv = convParams['sameConv']
        params.convolution_parameters.bias = convParams['bias']
        params.convolution_parameters.stride = convParams['stride']
        params.convolution_parameters.Kx = convParams['Kx']
        params.convolution_parameters.Ky = convParams['Ky']
        params.convolution_parameters.Noc = convParams['Noc']
        params.convolution_parameters.Nix = convParams['Nix']
        params.convolution_parameters.Niy = convParams['Niy']
        params.convolution_parameters.Nic = convParams['Nic']
        params.convolution_parameters.px_0 = convParams['px_0']
        params.convolution_parameters.px_1 = convParams['px_1']
        params.convolution_parameters.py_0 = convParams['py_0']
        params.convolution_parameters.py_1 = convParams['py_1']
        params.convolution_parameters.depthwise = convParams['depthwise']
        params.convolution_parameters.post_set() # This is done again later

    if NrowsMax is not None:
        params.xbar_params.NrowsMax = NrowsMax

    if On_off_ratio == 0:
        params.xbar_params.weights.minimum = 0
    else:
        params.xbar_params.weights.minimum = 1/On_off_ratio
    params.xbar_params.weights.maximum = 1
    params.xbar_params.weight_clipping.minimum = params.xbar_params.weights.minimum
    params.xbar_params.weight_clipping.maximum = params.xbar_params.weights.maximum

    params.algorithm_params.disable_clipping = False
    params.xbar_params.balanced_style = balanced_style

    # Read noise
    params.weight_error_params.noise_model = noise_model
    params.numeric_params.read_noise.sigma = alpha_noise

    # Proportional read noise or write error (default is additive)
    params.numeric_params.read_noise.proportional = proportional_noise

    # Programming error
    params.weight_error_params.proportional = proportional_error
    params.weight_error_params.error_model = error_model

    if params.weight_error_params.error_model == "alpha":
        # Apply a weight-independent programming error like read noise
        params.weight_error_params.sigma_error = alpha_error

    # Current drift and state-dependent write error
    if t_drift > 0 and drift_model != "none":
        params.weight_error_params.T = t_drift
        params.weight_error_params.drift_model = drift_model

    # Parasitic resistance
    if Rp > 0:
        # Bit line parasitic resistance
        params.numeric_params.Rp = Rp/params.xbar_params.weights.maximum
        params.numeric_params.Nex_par = x_par * y_par
        params.numeric_params.Niters_max_parasitics = 100
        params.numeric_params.circuit.noRowParasitics = noRowParasitics
        params.numeric_params.circuit.Vselect = 0
        params.numeric_params.circuit.Vread = 0.1 # Not relevant if Vselect=0
        # Under-relaxation
        params.numeric_params.Verr_th_mvm = 2e-4
        params.numeric_params.convergence_param = 0.9

        # Backprop only
        params.numeric_params.parasitic_backprop = False
        params.numeric_params.Verr_th_opu = 1e-3
        params.numeric_params.circuit.VrowS = 1.0
        params.numeric_params.circuit.VrowUS = -0.3
        params.numeric_params.circuit.VcolUS = 0.5
        params.numeric_params.circuit.Vprog = 0.1333
        params.numeric_params.convergence_param_opu = 0.5

    # Resolution on weights: quantization is applied during import
    # For BALANCED core, non-bitsliced, weight_bits is reduced by 1 since each device encodes half the range
    # This convention is not followed in the bit sliced case
    # As of 7/21/2021, this is accounted for here
    if wtmodel == "BALANCED" and Nslices == 1:
        weight_bits -= 1

    params.algorithm_params.weight_bits = weight_bits

    params.xbar_params.input_bitslicing = input_bitslicing
    params.xbar_params.ADC_per_ibit = ADC_per_ibit
    params.xbar_params.interleaved_posneg = interleaved_posneg
    params.xbar_params.fast_balanced = fast_balanced
    params.xbar_params.profile_ADC_inputs = profile_ADC_inputs

    params.xbar_params.adc_range_option = adc_range_option

    if Nslices == 1:
        # If ADC is applied after each input bit, the ADC ranges provided cannot be in algorithmic units
        # Instead these are in xbar units and applied directly to xbar values
        if ADC_per_ibit:
            params.xbar_params.adc_range_internal = adc_range
        if wtmodel == "OFFSET":
            params.xbar_params.offset_inference = True
            params.xbar_params.digital_offset = digital_offset

        # Dummy values: these limits will not actually be used
        if adc_range_option != "calibrated":
            adc_range = np.array([-1000000,1000000])

    else:
        params.xbar_params.Nslices = Nslices
        params.xbar_params.balanced_bitsliced = (wtmodel == "BALANCED")
        params.xbar_params.Nbits_reduction = adc_range
        if noRpSlices is not None:
            params.xbar_params.noRpSlices = noRpSlices
        else:
            params.xbar_params.noRpSlices = np.zeros(Nslices)

        # Since adc_range is not in algorithmic units or is not used, change it to a dummy value here
        # to prevent division by zero errors
        if adc_bits > 0:
            adc_range = np.array([-1000000,1000000])

    params.xbar_params.clip_Icol = (Icol_max > 0)
    params.xbar_params.Icol_max = Icol_max

    ####
    ### ADC and DAC settings
    ####

    # Forward prop
    params.xbar_params.col_input.bits = dac_bits
    params.xbar_params.row_output.bits = adc_bits

    # ADC/DAC ranges
    if dac_bits > 0:
        if input_bitslicing and not positiveInputsOnly:
            max_dac_range = np.max(np.abs(dac_range))
            params.algorithm_params.col_input.minimum = -max_dac_range
            params.algorithm_params.col_input.maximum = max_dac_range
        else:
            params.algorithm_params.col_input.minimum = dac_range[0]
            params.algorithm_params.col_input.maximum = dac_range[1]
    else:
        params.algorithm_params.col_input.minimum = -10000000000
        params.algorithm_params.col_input.maximum = 10000000000

    # Sign bit on input
    if positiveInputsOnly:
        params.xbar_params.col_input.sign_bit = False
        params.xbar_params.col_input.minimum = 0
    else:
        params.xbar_params.col_input.sign_bit = True

    # Sign bit on output
    # Sign bit on output means # levels is odd, ensuring zero will be one of the levels if the range is symmetric about zero
    # It should be used when the range is symmetric about zero, which is in one of the five cases below
    #   1) Bit slicing and differential
    #   2) Bit slicing and offset and negative inputs allowed
    #   3) Non bit slicing, calibrated, and ADC limits are very close to symmetric about zero
    #   4) Non bit slicing, differential and max/granular
    #   5) Non bit slicing, offset, max/granular, and negative inputs allowed
    if adc_bits > 0:
        if Nslices > 1:
            if wtmodel == "BALANCED":
                params.xbar_params.row_output.sign_bit = True
            elif wtmodel == "OFFSET" and not positiveInputsOnly:
                params.xbar_params.row_output.sign_bit = True
            else:
                params.xbar_params.row_output.sign_bit = False
        else:
            # This criterion checks if the center point of the range is within 1 level of zero
            # If that is the case, the range is made symmetric about zero and sign bit is used
            if adc_range_option == "calibrated":
                params.algorithm_params.row_output.minimum = adc_range[0]
                params.algorithm_params.row_output.maximum = adc_range[1]
                if np.abs(0.5*(adc_range[0] + adc_range[1])/(adc_range[1] - adc_range[0])) < 1/pow(2,adc_bits):
                    if np.abs(adc_range[0]) > np.abs(adc_range[1]):
                        adc_range[1] = -adc_range[0]
                    else:
                        adc_range[0] = -adc_range[1]
                    params.algorithm_params.row_output.minimum = adc_range[0]
                    params.algorithm_params.row_output.maximum = adc_range[1]
                    params.xbar_params.row_output.sign_bit = True
                else:
                    params.xbar_params.row_output.sign_bit = False
            elif adc_range_option == "max" or adc_range_option == "granular":
                if wtmodel == "BALANCED":
                    params.xbar_params.row_output.sign_bit = True
                elif wtmodel == "OFFSET" and not positiveInputsOnly:
                    params.xbar_params.row_output.sign_bit = True
                else:
                    params.xbar_params.row_output.sign_bit = False
    else:
        params.algorithm_params.row_output.minimum = -10000000000
        params.algorithm_params.row_output.maximum = 10000000000


    # Back prop settings (irrelevant for inference)
    params.xbar_params.row_input.bits = 0
    params.xbar_params.col_output.bits = 0
    params.xbar_params.col_output.sign_bit = True
    params.algorithm_params.row_input.minimum = -100000
    params.algorithm_params.row_input.maximum = 100000
    params.xbar_params.row_input.bits = 0
    params.algorithm_params.col_output.minimum = -100000
    params.algorithm_params.col_output.maximum = 100000
    params.xbar_params.col_output.bits = 0

    # Suppress the irrelevant warning about max update size
    params.algorithm_params.row_update.minimum = 0
    params.algorithm_params.row_update.maximum = 0
    params.algorithm_params.col_update.minimum = 0
    params.algorithm_params.col_update.maximum = 0

    return params


def inference(ntest,dataset,paramsList,sizes,keras_model,layerParams,**kwargs):
    """
    Runs inference on a full neural network whose weights are passed in as a Keras model and whose topology is specified by layerParams
    Parameters for individual neural cores generated using set_params() are passed in as paramsList
    """
    ########## load values from dict
    seed = random.randrange(1,1000000)
    
    count_interval = kwargs.get("count_interval",10)
    time_interval = kwargs.get("time_interval",False)
    randomSampling = kwargs.get("randomSampling",False)
    weight_percentile = kwargs.get("weight_percentile",100)
    dataset_noise_level = kwargs.get("dataset_noise_level",0.0)
    dataset_noise_type = kwargs.get("dataset_noise_type","gauss")
    mlp_style = kwargs.get("mlp_style",False)
    topk = kwargs.get("topk",1)
    nstart = kwargs.get("nstart",0)
    ntest_batch = kwargs.get("ntest_batch",0)
    calibration = kwargs.get("calibration",False)
    imagenet_preprocess = kwargs.get("imagenet_preprocess","cv2")
    dataset_normalization = kwargs.get("dataset_normalization","none")
    gpu_id = kwargs.get("gpu_id",0)
    useGPU = kwargs.get("useGPU",False)
    
    whetstone = kwargs.get("whetstone",False)
    larq = kwargs.get("larq",False)

    memory_window = kwargs.get("memory_window",0)
    subtract_pixel_mean = kwargs.get("subtract_pixel_mean",False)
    profiling_folder = kwargs.get("profiling_folder",None)
    profiling_settings = kwargs.get("profiling_settings",[False,False,False])
    model_name = kwargs.get("model_name",None)

    fold_batchnorm = kwargs.get("fold_batchnorm",False)
    batchnorm_style = kwargs.get("batchnorm_style","sqrt") # sqrt / no_sqrt
    digital_bias = kwargs.get("digital_bias",False)
    bias_bits = kwargs.get("bias_bits",0)
    ####################

    Nlayers = len(paramsList)
    bp = Backprop(sizes, seed=seed)
    bp.layerTypes = [layerParams[k]['type'] for k in range(len(layerParams))]
    
    sourceLayers = [layerParams[k]['source'] for k in range(len(layerParams))]
    if sourceLayers is not None:
        bp.sourceLayers = sourceLayers
        bp.memory_window = memory_window
    bp.alpha = 0
    bp.mlp_style = mlp_style
    bp.whetstone = whetstone
    bp.batchnorm_style = batchnorm_style

    # GPU is used either in none or all of the layers
    bp.init_GPU(useGPU,gpu_id)

    for k in range(Nlayers):
        if k < Nlayers-1:
            if layerParams[k]['activation'] is None:
                bp.set_activations(layer=k,style="NONE")
            elif layerParams[k]['activation']['type'] == "RECTLINEAR":
                bp.set_activations(layer=k,style=layerParams[k]['activation']['type'],relu_bound=layerParams[k]['activation']['bound'])
            elif layerParams[k]['activation']['type'] == "QUANTIZED_RELU":
                bp.set_activations(layer=k,style=layerParams[k]['activation']['type'],nbits=layerParams[k]['activation']['nbits'])
            elif layerParams[k]['activation']['type'] == "WHETSTONE":
                bp.set_activations(layer=k,style=layerParams[k]['activation']['type'],sharpness=layerParams[k]['activation']['sharpness'])
            else:
                bp.set_activations(layer=k,style=layerParams[k]['activation']['type'])
        else:
            if layerParams[k]['activation'] is None:
                bp.set_activate_output(style="NONE")
            elif layerParams[k]['activation']['type'] == "RECTLINEAR":
                bp.set_activate_output(style=layerParams[k]['activation']['type'],relu_bound=layerParams[k]['activation']['bound'])
            elif layerParams[k]['activation']['type'] == "QUANTIZED_RELU":
                bp.set_activate_output(style=layerParams[k]['activation']['type'],nbits=layerParams[k]['activation']['nbits'])
            elif layerParams[k]['activation']['type'] == "WHETSTONE":
                bp.set_activate_output(style=layerParams[k]['activation']['type'],sharpness=layerParams[k]['activation']['sharpness'])
            else:
                bp.set_activate_output(style=layerParams[k]['activation']['type'])

    print("Initializing neural cores")

    weight_dict = dict([(layer.name, layer.get_weights()) for layer in keras_model.layers])

    for m in range(Nlayers):
        params_m = paramsList[m]
        bp.auxLayerParams[m] = layerParams[m]

        if layerParams[m]['type'] not in ("conv","dense"):
            continue

        # Split MVM case
        if type(params_m) is list:
            Ncores = len(params_m)
        else:
            Ncores = 1

        Wm_0 = weight_dict[layerParams[m]['name']]
        Wm = Wm_0[0]
        if layerParams[m]['bias']:
            Wbias = Wm_0[1]

        # Quantize weights here if model is Larq
        if layerParams[m]['binarizeWeights']:
            Wm = np.sign(Wm)

        if fold_batchnorm and layerParams[m]['batch_norm'] is not None:
            if layerParams[m]['BN_scale'] and layerParams[m]['BN_center']:
                gamma, beta, mu, var = weight_dict[layerParams[m]['batch_norm']]
            elif layerParams[m]['BN_scale'] and not layerParams[m]['BN_center']:
                gamma, mu, var = weight_dict[layerParams[m]['batch_norm']]
                beta = 0
            elif not layerParams[m]['BN_scale'] and layerParams[m]['BN_center']:
                beta, mu, var = weight_dict[layerParams[m]['batch_norm']]
                gamma = 1
            else:
                mu, var = weight_dict[layerParams[m]['batch_norm']]
                gamma, beta = 1, 0
            
            epsilon = layerParams[m]['epsilon']
            if not layerParams[m]['bias']:
                Wbias = np.zeros(Wm.shape[-1])
                layerParams[m]['bias'] = True

            if batchnorm_style == "sqrt":
                if not (layerParams[m]['type'] == 'conv' and layerParams[m]['depthwise']):
                    Wm = gamma*Wm/np.sqrt(var + epsilon)
                    Wbias = (gamma/np.sqrt(var + epsilon))*(Wbias-mu) + beta
                else:
                    Wm = gamma[None,None,:,None]*Wm/np.sqrt(var[None,None,:,None] + epsilon)
                    Wbias = (gamma[None,None,:,None]/np.sqrt(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                    Wbias = np.squeeze(Wbias)

            elif batchnorm_style == "no_sqrt":
                if not (layerParams[m]['type'] == 'conv' and layerParams[m]['depthwise']):
                    Wm = gamma*Wm/(var + epsilon)
                    Wbias = (gamma/(var + epsilon))*(Wbias-mu) + beta
                else:
                    Wm = gamma[None,None,:,None]*Wm/(var[None,None,:,None] + epsilon)
                    Wbias = (gamma[None,None,:,None]/(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                    Wbias = np.squeeze(Wbias)

            weight_dict[layerParams[m]['name']] = [Wm,Wbias]

        if weight_percentile < 100:
            maxWeight_mPc = np.percentile(Wm,weight_percentile)
            minWeight_mPc = np.percentile(Wm,100-weight_percentile)
            maxWeight_m = np.max(np.abs((maxWeight_mPc,minWeight_mPc)))
            if layerParams[m]['bias'] and not digital_bias:
                maxWeight_biasPc = np.percentile(Wbias,weight_percentile)
                minWeight_biasPc = np.percentile(Wbias,100-weight_percentile)
                maxWeight_bias = np.max(np.abs((maxWeight_biasPc,minWeight_biasPc)))
            SF = 1
        else:
            maxWeight_m = np.max(np.abs(Wm))
            if layerParams[m]['bias'] and not digital_bias:
                maxWeight_bias = np.max(np.abs(Wbias))
            SF = weight_percentile/100

        if layerParams[m]['bias'] and not digital_bias:
            baseline_mat = SF*np.max((maxWeight_m,maxWeight_bias))
        else:
            baseline_mat = SF*maxWeight_m

        if Ncores == 1:
            params_m.algorithm_params.weights.maximum = baseline_mat
            params_m.algorithm_params.weights.minimum = -baseline_mat
        else:
            for k in range(Ncores):
                params_m[k].algorithm_params.weights.maximum = baseline_mat
                params_m[k].algorithm_params.weights.minimum = -baseline_mat
        # print("  Matrix "+str(m+1)+" Weight Limit = ",params_m.algorithm_params.weights.maximum)

        # set neural core parameters
        xbar_core_style = ("new_bias" if layerParams[m]['bias'] and not digital_bias else "new")
        if layerParams[m]['type'] == "conv":
            xbar_core_style = "conv"
            if digital_bias:
                if Ncores == 1:
                    params_m.convolution_parameters.bias = False
                else:
                    for k in range(Ncores):
                        params_m[k].convolution_parameters.bias = False

        # Set # images to profile
        if Ncores == 1 and params_m.xbar_params.profile_ADC_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m.xbar_params.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            params_m.xbar_params.Nimages_bitslicing = ntest
        elif Ncores > 1 and params_m[0].xbar_params.profile_ADC_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m[0].xbar_params.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            for k in range(Ncores):
                params_m[k].xbar_params.Nimages_bitslicing = ntest
                    
        bp.ncore(which=m+1, style=xbar_core_style, params=params_m)
        bp.digital_bias[m] = (digital_bias and layerParams[m]['bias'])

    # Import weights to CrossSim cores
    bp.read_weights_keras(weight_dict,layerParams,fold_batchnorm=fold_batchnorm)
    bp.expand_cores()

    # Import bias weights to be added digitally
    if digital_bias:
        bp.import_digital_bias(weight_dict,bias_bits,layerParams)

    # If using int4 model, import quantization and scale factors
    if model_name == "Resnet50-int4":
        bp.import_quantization(weight_dict,layerParams)
        bp.import_scale(weight_dict,layerParams)

    if keras_model is not None:
        del Wm, weight_dict, keras_model
        K.clear_session()
    else:
        del Wm

    if randomSampling:
        ntest_batch = ntest
        print("Warning: ntest_batch is ignored with random sampling")

    if ntest_batch > 0:
        if ntest_batch > ntest:
            ntest_batch = ntest
        nloads = (ntest-1) // ntest_batch + 1
        if type(topk) is int:
            frac_accum = 0
        else:
            frac_accum = np.zeros(len(topk))
    else:
        print('Loading dataset')
        ntest_batch = ntest
        nloads = 1
    nstart_i = nstart

    for nl in range(nloads):

        nend_i = nstart_i + ntest_batch

        ## Load truncated data set
        if nloads > 1:
            print('Loading dataset, images '+str(nl*ntest_batch)+' to '+str((nl+1)*ntest_batch)+' of '+str(ntest))

        if dataset == "mnist":
            if ntest_batch > 10000:
                raise ValueError("At most 10,000 test images can be used for MNIST")
            (x_test, y_test) = load_data_mnist(nstart=nstart_i,nend=nend_i,calibration=calibration)
        elif dataset == "fashion":
            if ntest_batch > 10000:
                raise ValueError("At most 10,000 test images can be used for Fashion MNIST")
            (x_test, y_test) = load_data_fashion_mnist(nstart=nstart_i,nend=nend_i,calibration=calibration)
        elif dataset == "cifar10":
            if ntest_batch > 10000:
                raise ValueError("At most 10,000 test images can be used for CIFAR-10")
            (x_test, y_test) = load_cifar_10(nstart=nstart_i,nend=nend_i,calibration=calibration,subtract_pixel_mean=subtract_pixel_mean)
        elif dataset == "cifar100":
            if ntest_batch > 10000:
                raise ValueError("At most 10,000 test images can be used for CIFAR-100")
            (x_test, y_test) = load_cifar_100(nstart=nstart_i,nend=nend_i,calibration=calibration,subtract_pixel_mean=subtract_pixel_mean)
        elif dataset == "imagenet":
            if ntest_batch > 50000:
                raise ValueError("At most 50,000 test images can be used for ImageNet")
            (x_test, y_test) = load_imagenet(option=imagenet_preprocess,nstart=nstart_i,nend=nend_i,calibration=calibration)
        else:
            raise ValueError("unknown dataset")

        x_test = x_test.astype(np.float32)

        # Dataset scaling
        # Add new cases here if this is not true
        if dataset_normalization == "unsigned_8b":
            x_test /= 255
        elif dataset_normalization == "signed_8b":
            x_test = x_test / 127.5 - 1

        # Apply noise to the dataset
        if dataset_noise_level > 0.0:
            if dataset_noise_type == "s&p":
                from skimage.util import random_noise
                x_test_noise = random_noise(x_test,mode='s&p',amount=dataset_noise_level)
            elif dataset_noise_type == "gauss":
                std_neu = dataset_noise_level**0.5
                gauss = np.random.normal(0,std_neu,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
                x_test_noise = x_test + gauss
            elif dataset_noise_type == "speckle":
                std_neu = dataset_noise_level**0.5
                gauss = np.random.normal(0,std_neu,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
                x_test_noise = x_test + x_test*gauss
            bp.indata = x_test_noise
        else:
            bp.indata = x_test
        bp.answers = y_test

        # If first layer is convolution, transpose the dataset so channel comes first
        if type(paramsList[0])==list:
            if paramsList[0][0].convolution_parameters.is_conv_core:
                bp.indata = np.transpose(bp.indata,(0,3,1,2))
        else:
            if paramsList[0].convolution_parameters.is_conv_core:
                bp.indata = np.transpose(bp.indata,(0,3,1,2))
        bp.ndata = (ntest_batch if not randomSampling else x_test.shape[0])

        # If the first layer is using GPU, send the inputs to the GPU
        if useGPU:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
            bp.indata = cp.array(bp.indata)

        # Run inference
        print("Beginning inference. Truncated test set to %d examples" % ntest)
        time_start = time.time()
        count, frac = bp.classify(n=ntest_batch,count_interval=count_interval,time_interval=time_interval,randomSampling=randomSampling,topk=topk,\
            profiling_folder=profiling_folder,profiling_settings=profiling_settings)

        # Print accuracy results
        time_stop = time.time()
        cpu = time_stop - time_start
        print("Total CPU seconds = %g" % cpu)
        if type(topk) is int:
            print("Inference accuracy: {:.3f}".format(frac*100)+"% ("+str(int(count))+"/"+str(ntest_batch)+")")
        else:
            accs = ""
            for j in range(len(topk)):
                accs += "{:.2f}".format(100*frac[j]) + "% (top-" + str(int(topk[j])) + ", "+str(int(count[j]))+"/"+str(ntest_batch)+")"
                if j < (len(topk)-1): accs += ", "
            print("Inference acuracy: "+accs)
        print("\n")

        if nloads > 1:
            nstart_i += ntest_batch
            frac_accum += frac

    del bp

    if nloads > 1:
        print("===========================")
        frac = frac_accum / nloads
        if type(topk) is int:
            print("Total inference accuracy: {:.2f}".format(100*frac)+"%")
        else:
            accs = ""
            for j in range(len(topk)):
                accs += "{:.2f}".format(100*frac[j]) + "% (top-" + str(topk[j]) + ")"
                if j < (len(topk)-1): accs += ", "
            print("Total inference acuracy: "+accs)
        print("===========================")
        print("\n")

    return frac