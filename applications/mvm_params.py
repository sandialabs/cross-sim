#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys
import numpy as np
sys.path.append("..")
from simulator import CrossSimParameters

def set_params(**kwargs):
    """
    Pass parameters using kwargs to allow for a general parameter dict to be used
    This function should be called before train and sets all parameters of the neural core simulator

    :return: params, a parameter object with all the settings

    """
    #######################

    #### load relevant parameters from arg dict
    wtmodel = kwargs.get("wtmodel","BALANCED")
    complex_input = kwargs.get("complex_input",False)
    complex_matrix = kwargs.get("complex_matrix",False)

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
    Rmin = kwargs.get("Rmin", 1000)
    Rmax = kwargs.get("Rmax", 10000)
    infinite_on_off_ratio = kwargs.get("infinite_on_off_ratio", False)

    NrowsMax = kwargs.get("NrowsMax",None)
    NcolsMax = kwargs.get("NcolsMax",None)
    weight_bits = kwargs.get("weight_bits",0)
    weight_percentile = kwargs.get("weight_percentile",100)
    input_bits = kwargs.get("input_bits",0)
    adc_bits = kwargs.get("adc_bits",0)
    input_range = kwargs.get("input_range",(-1e12,1e12))
    adc_range = kwargs.get("adc_range",(-1e12,1e12))
    adc_type = kwargs.get("adc_type","generic")
    positiveInputsOnly = kwargs.get("positiveInputsOnly",False)
    interleaved_posneg = kwargs.get("interleaved_posneg",False)
    subtract_current_in_xbar = kwargs.get("subtract_current_in_xbar",True)
    gate_input = kwargs.get("gate_input",False)

    Nslices = kwargs.get("Nslices",1)    
    digital_offset = kwargs.get("digital_offset",True)
    adc_range_option = kwargs.get("adc_range_option","CALIBRATED")
    Icol_max = kwargs.get("Icol_max",1e6)
    
    useGPU = kwargs.get("useGPU",False)
    gpu_id = kwargs.get("gpu_id",0)

    balanced_style = kwargs.get("balanced_style","ONE_SIDED")
    input_bitslicing = kwargs.get("input_bitslicing",False)
    input_slice_size = kwargs.get("input_slice_size",1)
    ADC_per_ibit = kwargs.get("ADC_per_ibit",False)
    disable_parasitics_slices = kwargs.get("disable_parasitics_slices",None)

    ################  create parameter objects with all core settings
    params = CrossSimParameters()

    params.core.complex_input = complex_input
    params.core.complex_matrix = complex_matrix

    ############### Numerical simulation settings
    params.simulation.useGPU = useGPU
    if useGPU:
        params.simulation.gpu_id = gpu_id

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

    ########### Device errors

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
    if drift_model and drift_model != "none":
        params.xbar.device.time = t_drift
        params.xbar.device.drift_error.model = drift_model

    ############### Parasitic resistance

    if Rp_col > 0 or Rp_row > 0:
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
    params.xbar.array.Icol_max = Icol_max

###################### DAC settings

    params.xbar.dac.mvm.bits = int(input_bits)
    if positiveInputsOnly or input_range[0] >= 0:
        params.xbar.dac.mvm.model = "QuantizerDAC"
        params.xbar.dac.mvm.signed = False
    else:
        params.xbar.dac.mvm.model = "SignMagnitudeDAC"
        params.xbar.dac.mvm.signed = True

    params.core.mapping.inputs.mvm.min = float(input_range[0])
    params.core.mapping.inputs.mvm.max = float(input_range[1])
    if input_bits > 0:
        params.xbar.dac.mvm.input_bitslicing = input_bitslicing
        params.xbar.dac.mvm.slice_size = input_slice_size
        if not positiveInputsOnly:        
            if input_bitslicing or params.xbar.dac.mvm.model == "SignMagnitudeDAC":
                max_input_range = float(np.max(np.abs(input_range)))
                params.core.mapping.inputs.mvm.min = -max_input_range
                params.core.mapping.inputs.mvm.max = max_input_range
    else:
        params.xbar.dac.mvm.input_bitslicing = False

    ###################### ADC settings

    params.xbar.adc.mvm.bits = int(adc_bits)

    if adc_type == "ramp":
        params.xbar.adc.mvm.model = "RampADC"
        params.xbar.adc.mvm.gain_db = 60
        params.xbar.adc.mvm.sigma_capacitor = 0.25
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.symmetric_cdac = True
    elif adc_type == "sar" or adc_type == "SAR":
        params.xbar.adc.mvm.model = "SarADC"
        params.xbar.adc.mvm.gain_db = 60
        params.xbar.adc.mvm.sigma_capacitor = 0.10
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.split_cdac = True
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "pipeline":
        params.xbar.adc.mvm.model = "PipelineADC"
        params.xbar.adc.mvm.gain_db = 60
        params.xbar.adc.mvm.sigma_C1 = 0.10
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.10
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "cyclic":
        params.xbar.adc.mvm.model = "CyclicADC"
        params.xbar.adc.mvm.gain_db = 60
        params.xbar.adc.mvm.sigma_C1 = 0.10
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.10
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