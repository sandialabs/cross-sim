#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

'''
Print a (non-exhaustive) list of simulation settings prior to the start of the
inference simulation
'''
def print_configuration_message(config):

    print("=======================================")
    print(config.task+" sim: "+str(config.ntest)+" images, start: " +str(config.nstart))
    print("Model: "+str(config.model_name))
    print("Mapping: "+str(config.style))
    if config.style == "BALANCED":
        if config.Nslices == 1:
            print('  Differential cells style: '+config.balanced_style)
        if config.interleaved_posneg:
            print('  Positive/negative weight cells interleaved on column')
        else:
            print('  Subtract current in crossbar: '+str(config.subtract_current_in_xbar))
    if config.style == "OFFSET":
        print('  Digital offset: '+str(config.digital_offset))
    if config.weight_bits > 0:
        print('  Weight quantization: '+str(config.weight_bits)+' bits')
    else:
        print('  Weight quantization off')
    if config.Nslices > 1:
        if config.weight_bits % config.Nslices == 0:
            Wbits_slice = int(config.weight_bits / config.Nslices)
        elif config.style == "BALANCED":
            Wbits_slice = np.ceil((config.weight_bits-1) / config.Nslices).astype(int)
        else:
            Wbits_slice = np.ceil(config.weight_bits / config.Nslices).astype(int)
        print('  Bit sliced: '+str(config.Nslices) + ' slices, '+str(Wbits_slice)+' bits/cell')
    else:
        if config.weight_bits == 0:
            device_bits = "infinite"
        elif config.style == "BALANCED":
            if config.model_name == "MobilenetV1-int8":
                device_bits = config.weight_bits
            elif config.model_name == "Resnet50-int4":
                device_bits = "variable, 4-7"
            else:
                device_bits = config.weight_bits - 1
        else:
            if config.model_name == "Resnet50-int4":
                device_bits = "variable, 4-8"
            else:
                device_bits = config.weight_bits
        print('  Bit slicing off, '+str(device_bits)+' bits/cell')
    print('  Digital bias: '+str(config.digital_bias))
    print('  Batchnorm fold: '+str(config.fold_batchnorm))
    if config.bias_bits == "adc":
        print('  Bias quantization: following ADC')
    elif config.bias_bits > 0:
        print('  Bias quantization: '+str(config.bias_bits)+' bits')
    else:
        print('  Bias quantization off')
    if config.NrowsMax > 0:
        print('  Max # rows: '+str(config.NrowsMax))
    else:
        print('  Unlimited # rows')
    if config.NcolsMax > 0:
        print('  Max # cols: '+str(config.NcolsMax))
    else:
        print('  Unlimited # cols')
    if config.error_model != 'none' and config.error_model != 'generic':
        print('  Weight error on ('+config.error_model+')')
    elif config.error_model == 'generic':
        if config.proportional_error:
            print('  Programming error (proportional): {:.3f}'.format(100*config.alpha_error)+'%')
        else:
            print('  Programming error (uniform): {:.3f}'.format(100*config.alpha_error)+'%')
    else:
        print('  Programming error off')
    if config.noise_model != "none":
        if config.noise_model == "generic":
            if config.alpha_noise > 0:
                if config.proportional_noise:
                    print('  Read noise (proportional): {:.3f}'.format(100*config.alpha_noise)+'%')
                else:
                    print('  Read noise (uniform): {:.3f}'.format(100*config.alpha_noise)+'%')
            else:
                print('  Read noise off')
        else:
            print('  Read noise on ('+config.noise_model+')')
    else:
        print('  Read noise off')
    if config.adc_bits > 0:
        if np.min(config.adc_bits_vec) == np.max(config.adc_bits_vec):
            print('  ADC on, '+str(config.adc_bits)+' bits')
        else:
            print('  ADC on, variable: '+str(np.min(config.adc_bits_vec))+'-'+str(np.max(config.adc_bits_vec))+' bits')
        if config.ADC_per_ibit:
            print('    ADC per input bit: ON')
        print('    ADC range option: '+config.adc_range_option)
        if config.Nslices > 1 and config.adc_range_option == "calibrated":
            print('    Bit sliced ADC range calibration percentile: '+str(config.pct)+'%')
        print('    ADC topology: '+str(config.adc_type))
    else:
        print('  ADC off')
    if config.dac_bits > 0:
        if np.min(config.dac_bits_vec) == np.max(config.dac_bits_vec):
            print('  Activation quantization on, '+str(config.dac_bits)+' bits')
        else:
            print('  Activation quantization on, variable: '+str(np.min(config.dac_bits_vec))+'-'+str(np.max(config.dac_bits_vec))+' bits')
        print('  Input bit slicing: '+str(config.input_bitslicing))
        if config.input_bitslicing:
            print('     Input slice size: '+str(config.input_slice_size)+" bits")
    else:
        print('  Activation quantization off')
    if config.Icol_max > 0:
        print('  Column current clipping on: {:.3f}'.format(config.Icol_max*1e6)+" uA")
    if config.Rp_col > 0:
        print('  Rp column = {:.3e} (ohms)'.format(config.Rp_col))
    if config.Rp_row > 0 and not config.gate_input:
        print('  Rp row = {:.3e} (ohms)'.format(config.Rp_row))
    if config.Rp_col == 0 and (config.Rp_row == 0 or config.gate_input):
        print('  Parasitics off')
    if config.Rp_col > 0 or config.Rp_row > 0:
        print('  Gate input mode: '+str(config.gate_input))
    if config.infinite_on_off_ratio:
        print('  On off ratio: infinite')
    else:
        print('  On off ratio: {:.1f}'.format(config.Rmax/config.Rmin))
    if config.t_drift > 0 or config.drift_model != "none":
        print('  Weight drift on, '+str(config.t_drift)+' days')
        print('  Drift model: '+config.drift_model)
    else:
        print('  Weight drift off')
    if config.useGPU:
        print('  GPU: '+str(config.gpu_num))
    print("=======================================")
