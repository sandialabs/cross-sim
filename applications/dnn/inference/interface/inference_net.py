#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys, io, os, time, random
import numpy as np
import tensorflow.keras.backend as K
from simulator import DNN, CrossSimParameters
from dataset_loaders import load_dataset_inference
from helpers.qnn_adjustment import qnn_adjustment
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
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

    Rp_row = kwargs.get("Rp_row",0)
    Rp_col = kwargs.get("Rp_col",0)

    NrowsMax = kwargs.get("NrowsMax",None)
    NcolsMax = kwargs.get("NcolsMax",None)
    weight_bits = kwargs.get("weight_bits",0)
    weight_percentile = kwargs.get("weight_percentile",100)
    adc_bits = kwargs.get("adc_bits",8)
    dac_bits = kwargs.get("dac_bits",8)
    dac_range = kwargs.get("dac_range",(0,1))
    adc_range = kwargs.get("adc_range",(0,1))
    adc_type = kwargs.get("adc_type","generic")
    positiveInputsOnly = kwargs.get("positiveInputsOnly",False)
    interleaved_posneg = kwargs.get("interleaved_posneg",False)
    subtract_current_in_xbar = kwargs.get("subtract_current_in_xbar",True)
    Rmin = kwargs.get("Rmin", 1000)
    Rmax = kwargs.get("Rmax", 10000)
    infinite_on_off_ratio = kwargs.get("infinite_on_off_ratio", True)
    gate_input = kwargs.get("gate_input",True)

    Nslices = kwargs.get("Nslices",1)    
    digital_offset = kwargs.get("digital_offset",False)
    adc_range_option = kwargs.get("adc_range_option",False)
    Icol_max = kwargs.get("Icol_max",1e6)
    digital_bias = kwargs.get("digital_bias",False)
    analog_batchnorm = kwargs.get("analog_batchnorm",False)

    x_par = kwargs.get("x_par",1)
    y_par = kwargs.get("y_par",1)
    weight_reorder = kwargs.get("weight_reorder",True)
    conv_matmul = kwargs.get("conv_matmul",True)
    useGPU = kwargs.get("useGPU",False)
    gpu_id = kwargs.get("gpu_id",0)
    profile_ADC_inputs = kwargs.get("profile_ADC_inputs",False)
    profile_ADC_reluAware = kwargs.get("profile_ADC_reluAware",False)

    balanced_style = kwargs.get("balanced_style","one_sided")
    input_bitslicing = kwargs.get("input_bitslicing",False)
    input_slice_size = kwargs.get("input_slice_size",1)
    ADC_per_ibit = kwargs.get("ADC_per_ibit",False)
    disable_parasitics_slices = kwargs.get("disable_parasitics_slices",None)

    ################  create parameter objects with all core settings
    params = CrossSimParameters()

    ############### Numerical simulation settings
    params.simulation.useGPU = useGPU
    if useGPU:
        params.simulation.gpu_id = gpu_id

    ### Weight reoder: Whether to reorder the duplicated weights into Toeplitz form when x_par, y_par > 1 so that the
    # expanded matrix takes up less memory
    if convParams is None:
        weight_reorder = False
    else:
        # This option is disabled if any of the following conditions are True
        # 1-3) No reuse to exploit
        noWR_cond1 =    (x_par == 1 and y_par == 1)
        noWR_cond2 =    (convParams['Kx'] == 1 and convParams['Ky'] == 1)
        noWR_cond3 =    ((x_par > 1 and convParams['stride'] >= convParams['Kx']) and \
                        (y_par > 1 and convParams['stride'] >= convParams['Ky']))
        # 4-11) Cases that don't make sense to implement (as of now)
        noWR_cond4 =    (NrowsMax < convParams['Kx']*convParams['Ky']*convParams['Nic'])
        noWR_cond5 =    (convParams['bias'] and not digital_bias)
        noWR_cond6 =    (analog_batchnorm and not digital_bias)
        noWR_cond7 =    (wtmodel == "OFFSET")
        noWR_cond8 =    (Rp_col > 0)
        noWR_cond9 =    (Rp_row > 0 and not gate_input)
        noWR_cond10 =   (noise_model == "generic" and alpha_noise > 0)
        noWR_cond11 =   (noise_model != "none" and noise_model != "generic")

        if any([noWR_cond1, noWR_cond2, noWR_cond3, noWR_cond4, noWR_cond5, noWR_cond6, 
            noWR_cond7, noWR_cond8, noWR_cond9, noWR_cond10, noWR_cond11]):
            weight_reorder = False

    # Enable conv matmul?
    if convParams is None:
        conv_matmul = False
    else:
        # These cases cannot be realistically modeled with matmul
        noCM_cond1 = (Rp_col > 0)
        noCM_cond2 = (Rp_row > 0 and not gate_input)
        noCM_cond3 = (noise_model == "generic" and alpha_noise > 0)
        noCM_cond4 = (noise_model != "none" and noise_model != "generic")
        if any([noCM_cond1, noCM_cond2, noCM_cond3, noCM_cond4]):
            conv_matmul = False

    if conv_matmul:
        weight_reorder = False
        x_par = 1
        y_par = 1

    # Multiple convolutional MVMs in parallel? (only used if conv_matmul = False)
    params.simulation.convolution.x_par = int(x_par) # Number of sliding window steps to do in parallel (x)
    params.simulation.convolution.y_par = int(y_par) # Number of sliding window steps to do in parallel (y)
    params.simulation.convolution.weight_reorder = weight_reorder
    params.simulation.convolution.conv_matmul = conv_matmul

    ############### Crossbar weight mapping settings

    if Nslices == 1:
        params.core.style = wtmodel
    else:
        params.core.style = "BITSLICED"
        params.core.bit_sliced.style = wtmodel

    if NrowsMax is not None:
        params.core.rows_max = NrowsMax

    if NcolsMax is not None:
        params.core.cols_max = NcolsMax

    params.core.balanced.style = balanced_style
    params.core.balanced.subtract_current_in_xbar = subtract_current_in_xbar
    if digital_offset:
        params.core.offset.style = "DIGITAL_OFFSET"
    else:
        params.core.offset.style = "UNIT_COLUMN_SUBTRACTION"

    params.xbar.device.Rmin = Rmin
    params.xbar.device.Rmax = Rmax
    params.xbar.device.infinite_on_off_ratio = infinite_on_off_ratio

    ############### Convolution

    if convParams is not None:
        params.simulation.convolution.is_conv_core = (convParams['type'] == 'conv')
        params.simulation.convolution.stride = convParams['stride']
        params.simulation.convolution.Kx = convParams['Kx']
        params.simulation.convolution.Ky = convParams['Ky']
        params.simulation.convolution.Noc = convParams['Noc']
        params.simulation.convolution.Nic = convParams['Nic']

    ############### Device errors

    #### Read noise
    if noise_model == "generic" and alpha_noise > 0:
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.magnitude = alpha_noise
        if not proportional_noise:
            params.xbar.device.read_noise.model = "NormalIndependentDevice"
        else:
            params.xbar.device.read_noise.model = "NormalProportionalDevice"
    elif noise_model != "generic" and noise_model != "none":
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.model = noise_model

    ##### Programming error
    if error_model == "generic" and alpha_error > 0:
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.magnitude = alpha_error
        if not proportional_error:
            params.xbar.device.programming_error.model = "NormalIndependentDevice"
        else:
            params.xbar.device.programming_error.model = "NormalProportionalDevice"
    elif error_model != "generic" and error_model != "none":
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.model = error_model

    # Drift
    if drift_model != "none":
        params.xbar.device.drift_error.enable = True
        params.xbar.device.time = t_drift
        params.xbar.device.drift_error.model = drift_model

    ############### Parasitic resistance

    if Rp_col > 0 or Rp_row > 0:
        # Bit line parasitic resistance
        params.xbar.array.parasitics.enable = True
        params.xbar.array.parasitics.Rp_col = Rp_col/Rmin
        params.xbar.array.parasitics.Rp_row = Rp_row/Rmin
        params.xbar.array.parasitics.gate_input = gate_input

        if gate_input and Rp_col == 0:
            params.xbar.array.parasitics.enable = False

        # Numeric params related to parasitic resistance simulation
        params.simulation.Niters_max_parasitics = 100
        params.simulation.Verr_th_mvm = 2e-4
        params.simulation.relaxation_gamma = 0.9 # under-relaxation

    ############### Weight bit slicing

    # Compute the number of cell bits
    if wtmodel == "OFFSET" and Nslices == 1:
        cell_bits = weight_bits
    elif wtmodel == "BALANCED" and Nslices == 1:
        cell_bits = weight_bits - 1
    elif Nslices > 1:
        # For weight bit slicing, quantization is done during mapping and does
        # not need to be applied at the xbar level
        if weight_bits % Nslices == 0:
            cell_bits = int(weight_bits / Nslices)
        elif wtmodel == "BALANCED":
            cell_bits = int(np.ceil((weight_bits-1)/Nslices))
        else:
            cell_bits = int(np.ceil(weight_bits/Nslices))
        params.core.bit_sliced.num_slices = Nslices
        if disable_parasitics_slices is not None:
            params.xbar.array.parasitics.disable_slices = disable_parasitics_slices
        else:
            params.xbar.array.parasitics.disable_slices = [False]*Nslices

    # Weights
    params.core.weight_bits = int(weight_bits)
    params.core.mapping.weights.percentile = weight_percentile/100

    params.xbar.device.cell_bits = cell_bits
    params.xbar.adc.mvm.adc_per_ibit = ADC_per_ibit
    params.xbar.adc.mvm.adc_range_option = adc_range_option
    params.core.balanced.interleaved_posneg = interleaved_posneg
    params.simulation.analytics.profile_adc_inputs = (profile_ADC_inputs and not profile_ADC_reluAware)
    params.xbar.array.Icol_max = Icol_max

    ###################### DAC settings

    params.xbar.dac.mvm.bits = int(dac_bits)
    if positiveInputsOnly:
        params.xbar.dac.mvm.model = "QuantizerDAC"
        params.xbar.dac.mvm.signed = False
    else:
        params.xbar.dac.mvm.model = "SignMagnitudeDAC"
        params.xbar.dac.mvm.signed = True

    if dac_bits > 0:
        params.xbar.dac.mvm.input_bitslicing = input_bitslicing
        params.xbar.dac.mvm.slice_size = input_slice_size
        if not digital_bias:
            dac_range[1] = np.maximum(dac_range[1],1)

        if not positiveInputsOnly:        
            if input_bitslicing or params.xbar.dac.mvm.model == "SignMagnitudeDAC":
                max_dac_range = float(np.max(np.abs(dac_range)))
                params.core.mapping.inputs.mvm.min = -max_dac_range
                params.core.mapping.inputs.mvm.max = max_dac_range
            else:
                params.core.mapping.inputs.mvm.min = float(dac_range[0])
                params.core.mapping.inputs.mvm.max = float(dac_range[1])
        else:
            params.core.mapping.inputs.mvm.min = float(dac_range[0])
            params.core.mapping.inputs.mvm.max = float(dac_range[1])
    else:
        params.xbar.dac.mvm.input_bitslicing = False
        params.core.mapping.inputs.mvm.min = -1e10
        params.core.mapping.inputs.mvm.max = 1e10

    ###################### ADC settings

    params.xbar.adc.mvm.bits = int(adc_bits)

    # Custom ADCs are currently set to ideal. Modify the parameters
    # below to model non-ideal ADC implementations
    if adc_type == "ramp":
        params.xbar.adc.mvm.model = "RampADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.symmetric_cdac = True
    elif adc_type == "sar" or adc_type == "SAR":
        params.xbar.adc.mvm.model = "SarADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.split_cdac = True
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "pipeline":
        params.xbar.adc.mvm.model = "PipelineADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "cyclic":
        params.xbar.adc.mvm.model = "CyclicADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8

    # Determine if signed ADC
    if adc_range_option == "CALIBRATED" and adc_bits > 0:
        params.xbar.adc.mvm.signed = bool(np.min(adc_range) < 0)
    else:
        params.xbar.adc.mvm.signed = (wtmodel == "BALANCED" or (wtmodel == "OFFSET" and not positiveInputsOnly))

    # Set the ADC model and the calibrated range
    if adc_bits > 0:
        if Nslices > 1:
            if adc_range_option == "CALIBRATED":
                params.xbar.adc.mvm.calibrated_range = adc_range
            if params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "SignMagnitudeADC"
            elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "QuantizerADC"
        else:
            # This criterion checks if the center point of the range is within 1 level of zero
            # If that is the case, the range is made symmetric about zero and sign bit is used
            if adc_range_option == "CALIBRATED":
                if np.abs(0.5*(adc_range[0] + adc_range[1])/(adc_range[1] - adc_range[0])) < 1/pow(2,adc_bits):
                    absmax = np.max(np.abs(adc_range))
                    params.xbar.adc.mvm.calibrated_range = np.array([-absmax,absmax])
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "SignMagnitudeADC"
                else:
                    params.xbar.adc.mvm.calibrated_range = adc_range
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "QuantizerADC"

            elif adc_range_option == "MAX" or adc_range_option == "GRANULAR":
                if params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "SignMagnitudeADC"
                elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "QuantizerADC"
                    
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
    topk = kwargs.get("topk",1)
    nstart = kwargs.get("nstart",0)
    ntest_batch = kwargs.get("ntest_batch",0)
    calibration = kwargs.get("calibration",False)
    imagenet_preprocess = kwargs.get("imagenet_preprocess","cv2")
    dataset_normalization = kwargs.get("dataset_normalization","none")
    gpu_id = kwargs.get("gpu_id",0)
    useGPU = kwargs.get("useGPU",False)
    
    memory_window = kwargs.get("memory_window",0)
    subtract_pixel_mean = kwargs.get("subtract_pixel_mean",False)
    profiling_folder = kwargs.get("profiling_folder",None)
    profiling_settings = kwargs.get("profiling_settings",[False,False,False])
    model_name = kwargs.get("model_name",None)

    fold_batchnorm = kwargs.get("fold_batchnorm",False)
    batchnorm_style = kwargs.get("batchnorm_style","sqrt") # sqrt / no_sqrt
    digital_bias = kwargs.get("digital_bias",False)
    bias_bits = kwargs.get("bias_bits",0)
    show_HW_config = kwargs.get("show_HW_config",False)
    return_network_output = kwargs.get("return_network_output",False)

    ####################

    dnn = DNN(sizes, seed=seed)
    dnn.set_inference_params(layerParams,
        memory_window=memory_window,
        batchnorm_style=batchnorm_style,
        fold_batchnorm=fold_batchnorm)
    dnn.init_GPU(useGPU,gpu_id)
    
    Nlayers = len(paramsList)
    Nlayers_mvm = 0
    for k in range(Nlayers):
        if layerParams[k]['activation'] is None:
            dnn.set_activations(layer=k,style="NONE")
        elif layerParams[k]['activation']['type'] == "RECTLINEAR":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],relu_bound=layerParams[k]['activation']['bound'])
        elif layerParams[k]['activation']['type'] == "QUANTIZED_RELU":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],nbits=layerParams[k]['activation']['nbits'])
        elif layerParams[k]['activation']['type'] == "WHETSTONE":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],sharpness=layerParams[k]['activation']['sharpness'])
        else:
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'])

    print("Initializing neural cores")
    weight_dict = dict([(layer.name, layer.get_weights()) for layer in keras_model.layers])

    for m in range(Nlayers):
        params_m = paramsList[m]
        if layerParams[m]['type'] not in ("conv","dense"):
            dnn.set_layer_params(m, layerParams[m], digital_bias)
            continue
        Nlayers_mvm += 1

        Ncores = (len(params_m) if type(params_m) is list else 1)
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

        ### Explicit handling of quantized neural networks
        if model_name == "MobilenetV1-int8" or model_name == "Resnet50-int4":
            params_m = qnn_adjustment(model_name, params_m.copy(), Wm, Ncores, Nlayers_mvm)

        # set neural core parameters
        if digital_bias:
            layerParams[m]['bias_row'] = False
        else:
            layerParams[m]['bias_row'] = layerParams[m]['bias']

        if layerParams[m]['type'] == "conv":
            if Ncores == 1:
                params_m.simulation.convolution.bias_row = layerParams[m]['bias_row']
            else:
                for k in range(Ncores):
                    params_m[k].simulation.convolution.bias_row = layerParams[m]['bias_row']

        # Set # images to profile
        if Ncores == 1 and params_m.simulation.analytics.profile_adc_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m.xbar.dac.mvm.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            params_m.simulation.analytics.ntest = ntest
        elif Ncores > 1 and params_m[0].simulation.analytics.profile_adc_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m[0].xbar.dac.mvm.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            for k in range(Ncores):
                params_m[k].simulation.analytics.ntest = ntest

        dnn.set_layer_params(m, layerParams[m], digital_bias)
        dnn.ncore(m, style=layerParams[m]['type'], params=params_m)

    # Import weights to CrossSim cores
    dnn.read_weights_keras(weight_dict)
    dnn.expand_cores()

    # Import bias weights to be added digitally
    if digital_bias:
        dnn.import_digital_bias(weight_dict,bias_bits)

    # If using int4 model, import quantization and scale factors
    if model_name == "Resnet50-int4":
        dnn.import_quantization(weight_dict)
        dnn.import_scale(weight_dict)

    # Get a params object to use for easy access to simulation params
    params_0 = (paramsList[0][0] if type(paramsList[0])==list else paramsList[0])

    if keras_model is not None:
        del Wm, weight_dict, keras_model
        K.clear_session()
    else:
        del Wm

    if randomSampling:
        ntest_batch = ntest
        print("Warning: ntest_batch is ignored with random sampling")

    if show_HW_config:
        print("\n\n============================")
        print("Analog HW configuration")
        print("============================\n")
        dnn.show_HW_config()
        input("\nPress any key to continue...")

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
    network_outputs = None

    for nl in range(nloads):

        ## Load truncated data set
        nend_i = nstart_i + ntest_batch
        if nloads > 1:
            print('Loading dataset, images {:d} to {:d} of {:d}'.format(nl*ntest_batch,(nl+1)*ntest_batch,ntest))

        # Set FP precision based on the ADC/DAC precision: use 24 bits as cut-off
        if params_0.xbar.adc.mvm.bits > 24 or params_0.xbar.dac.mvm.bits > 24:
            precision = np.float64
        else:
            precision = np.float32

        # Load input data and labels
        (x_test, y_test) = load_dataset_inference(dataset, 
                                nstart_i, nend_i, 
                                calibration = calibration,
                                precision = precision,
                                subtract_pixel_mean = subtract_pixel_mean,
                                imagenet_preprocess = imagenet_preprocess,
                                dataset_normalization = dataset_normalization)

        # If first layer is convolution, transpose the dataset so channel comes first
        if params_0.simulation.convolution.is_conv_core:
            x_test = np.transpose(x_test,(0,3,1,2))

        dnn.indata = x_test
        dnn.answers = y_test
        dnn.ndata = (ntest_batch if not randomSampling else x_test.shape[0])

        # If the first layer is using GPU, send the inputs to the GPU
        if useGPU:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
            dnn.indata = cp.array(dnn.indata)

        # Run inference
        print("Beginning inference. Truncated test set to %d examples" % ntest)
        time_start = time.time()
        count, frac, network_output = dnn.predict(
                n = ntest_batch,
                count_interval = count_interval,
                time_interval = time_interval,
                randomSampling = randomSampling,
                topk = topk,
                return_network_output = return_network_output,
                profiling_folder = profiling_folder,
                profiling_settings = profiling_settings)

        # Print accuracy results
        time_stop = time.time()
        device = "CPU" if not useGPU else "GPU"
        sim_time = time_stop - time_start
        print("Total " + device + " seconds = {:.3f}".format(sim_time))
        if type(topk) is int:
            print("Inference accuracy: {:.3f}% ({:d}/{:d})".format(frac*100,count,ntest_batch))
        else:
            accs = ""
            for j in range(len(topk)):
                accs += "{:.2f}% (top-{:d}, {:d}/{:d})".format(100*frac[j], topk[j], count[j], ntest_batch)
                if j < (len(topk)-1): accs += ", "
            print("Inference acuracy: "+accs+"\n")

        # Consolidate neural network outputs
        if return_network_output:
            if nloads == 1:
                network_outputs = network_output
            else:
                if nl == 0:
                    network_outputs = np.zeros((ntest, network_output.shape[1]))
                network_outputs[(nl*ntest_batch):((nl+1)*ntest_batch),:] = network_output

        if nloads > 1:
            nstart_i += ntest_batch
            frac_accum += frac

    # Print overall neural network accuracy (all batches)
    if nloads > 1:
        print("===========================")
        frac = frac_accum / nloads
        if type(topk) is int:
            print("Total inference accuracy: {:.2f}%".format(100*frac))
        else:
            accs = ""
            for j in range(len(topk)):
                accs += "{:.2f}% (top-{:d})".format(100*frac[j],topk[j])
                if j < (len(topk)-1): accs += ", "
            print("Total inference acuracy: "+accs)
        print("===========================\n")

    return frac, network_outputs