import numpy as np

def find_adc_range(params_args, n_layers):
    """
    Given a set of arguments to dnn_inference_params(), find a compatible set
    of calibrated ADC ranges. If a calibrated range does not exist, this function
    will throw an error.
    """

    # Relevant inputs for selecting ADC and activation ranges
    adc_range_option = params_args['adc_range_option']
    adc_bits = params_args['adc_bits']
    style = params_args['core_style']
    NrowsMax = params_args['NrowsMax']
    adc_per_ibit = (params_args['input_bitslicing'] 
        and params_args['input_slice_size'] == 1
        and params_args['adc_per_ibit'])
    Nslices = params_args['Nslices']
    subtract_current_in_xbar = (params_args['subtract_current_in_xbar'] or 
        params_args['interleaved_posneg'])

    adc_ranges = [None] * n_layers

    # An NrowsMax value of None really means no upper bound
    if NrowsMax is None:
        NrowsMax = 1e6

    # The optimal ADC range depends on several factors
    #   - Number of weight bit slices
    #   - The negative weight handling scheme (style)
    #   - The array size (NrowsMax)
    #   - Whether input bit accumulation is done in analog (relevant for offset and bit slicing)
    #   - For balanced core, whether current is subtracted in analog (here, always assumed true)
 
    limits_dir = "./calibrated_config/"

    if adc_bits > 0 and adc_range_option == "CALIBRATED":

        if Nslices == 1:
            if style == "BALANCED" and NrowsMax >= 1152 and not adc_per_ibit and subtract_current_in_xbar:
                adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_balanced.npy")

            elif style == "OFFSET" and NrowsMax >= 1152 and adc_per_ibit:
                adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_offset.npy")

            elif style == "BALANCED" and NrowsMax == 144 and not adc_per_ibit and subtract_current_in_xbar:
                adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_balanced_144rows.npy")

            elif style == "BALANCED" and NrowsMax == 1152 and adc_per_ibit and subtract_current_in_xbar:
                adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_balanced_ibits.npy")

        else:
            try:
                if style == "BALANCED" and not adc_per_ibit and subtract_current_in_xbar:
                    adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_"+str(NrowsMax)+"rows_"+str(Nslices)+"slices_balanced.npy")
                elif style == "OFFSET" and adc_per_ibit:
                    adc_ranges = np.load(limits_dir+"adc_limits_ResNet50v15_"+str(NrowsMax)+"rows_"+str(Nslices)+"slices_offset.npy")
                else:
                    raise ValueError("No calibrated ADC range found for the combination of crossbar style, adc_per_ibit and # slices")
            except FileNotFoundError:
                raise ValueError("No calibrated ADC range found for the combination of crossbar style, adc_per_ibit and # slices")

        if adc_ranges[0] is None:
            raise ValueError("For the chosen crossbar settings, calibrated ADC ranges are unavailable for ResNet50-v1.5: "+\
                "please generate the limits, choose another range setting ('MAX' or 'GRANULAR'), or disable ADC.")

    return adc_ranges
