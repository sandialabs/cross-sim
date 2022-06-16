#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import os, pickle
from inference_util.keras_parser import load_keras_model



'''
augment_parameters:
    This is a helper function that checks the consistency of the parameters in config, and adds
    additional simulation parameters based on the ones that were set
    Returns: modified config object
'''
def augment_parameters(config):

    if config.Nslices == 1 or config.adc_range_option != "calibrated":
        config.pct = None

    # Whether ADC is applied after every input bit (i.e. input bits are accumulated after the ADC)
    if not config.input_bitslicing:
        config.ADC_per_ibit = False
        if config.style == "OFFSET" and config.adc_bits > 0:
            # Computation of the offset is practically much more complex if inputs are multi-bit
            # If inputs are single bit the offset is computed using a simple popcount
            print("Warning: OFFSET is used and ADC is enabled, but input bit slicing is not enabled.")

    if (config.adc_range_option != "calibrated" or config.Nslices > 1) and config.bias_bits == "adc":
        raise ValueError("Bias bits cannot track ADC resolution with the chosen ADC settings. Please set bias_bits to an integer")

    if config.Rp > 0 and config.noRowParasitics:
        if not config.input_bitslicing or config.dac_bits == 0:
            raise ValueError("If using no-row parasitics model, must enable input bit slicing")
    if config.Rp > 0 and config.dac_bits == 0:
            raise ValueError("For parasitic resistance simulations, must have dac_bits > 0 to define finite input voltage ranges."+\
                " If input ranges are available but input quantization is not desired, set dac_bits to a very large value (e.g. 32).")

    if config.Rp > 0 and not config.noRowParasitics and config.interleaved_posneg:
        print('Warning: interleaved pos/neg parasitic option ignored sice row parasitic resistance is enabled')
    if config.Rp > 0 and config.interleaved_posneg and config.style != "BALANCED":
        print('Warning: interleaved pos/neg parasitic option ignored sice differential cells are not used')

    if config.dac_bits == 0:
        config.input_bitslicing = False

    # By default, make every layer have the same ADC/DAC resolution
    # If desired, this can be changed on a model-wise basis (for example, see Resnet50-int4 in model_specific_parameters)
    config.adc_bits_vec = config.adc_bits * np.ones(config.Nlayers_mvm,dtype=int)
    config.dac_bits_vec = config.dac_bits * np.ones(config.Nlayers_mvm,dtype=int)

    if config.t_drift > 0 and config.drift_model != "none":
        if config.error_model != 'none' and config.error_model != 'alpha':
            print("Warning: custom device programming error model ignored since custom device drift model is active")
            config.error_model = 'none'

    # fast_balanced gives a performance boost with BALANCED or BITSLICED_BALANCED core if certain conditions are met
    noReadNoise = (config.noise_model == "alpha" and config.alpha_noise == 0) or config.noise_model == "none"
    if config.Rp > 0 or not noReadNoise or config.interleaved_posneg:
        config.fast_balanced = False
    else:
        config.fast_balanced = True

    return config

'''
build_keras_model:
Find the Keras model file associated with a model_name and build a Keras model object
If model_name does not map to one in the list, it attempts to use it as a filepath
   model_name  : name of the model, or path to Keras model
   show_model_summary  : print the Keras model summary
Returns: Keras model object
'''
def build_keras_model(model_name,show_model_summary=False):

    ##### IMAGENET MODELS
    if model_name == "Resnet50":
        # from keras.applications.resnet50 import ResNet50
        # keras_file = "../pretrained_models/imagenet/resnet50.h5"
        from tensorflow.keras.applications.resnet50 import ResNet50
        keras_file = "../pretrained_models/imagenet/resnet50.h5"
        keras_model = ResNet50(weights=keras_file)

    elif model_name == "Resnet50-v1.5":
        from helpers.keras_model_builders.buildResnet50v15 import ResNet50v15
        weights_file = "../pretrained_models/imagenet/MLperf/resnet50v15-weights.p"
        weights_dict = pickle.load(open(weights_file,"rb"))
        keras_model = ResNet50v15(weights_dict)

    elif model_name == "Resnet50-int4":
        from helpers.keras_model_builders.buildResnet50_int4 import ResNet50v15_int4
        weights_file = "../pretrained_models/imagenet/nvidia_mlperf_int4/resnet50_int4.p"
        quant_file = "../pretrained_models/imagenet/nvidia_mlperf_int4/resnet50_int4_q.p"
        weights_dict = pickle.load(open(weights_file,"rb"))
        quant_dict = pickle.load(open(quant_file,"rb"))
        keras_model = ResNet50v15_int4(weights_dict,quant_dict)

    elif model_name == "VGG19":
        from tensorflow.keras.applications import VGG19
        keras_model = VGG19(weights="imagenet")

    elif model_name == "MobilenetV2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        keras_file = "../pretrained_models/imagenet/mobilenetV2.h5"
        keras_model = MobileNetV2(weights=keras_file)

    elif model_name == "MobilenetV1":
        from tensorflow.keras.applications.mobilenet import MobileNet
        keras_file = "../pretrained_models/imagenet/mobilenetV1.h5"
        keras_model = MobileNet(weights=keras_file)

    elif model_name == "MobilenetV1-int8":
        from helpers.keras_model_builders.buildMobilenet_int8 import MobileNet_int8
        weights_file = "../pretrained_models/imagenet/MLperf/mobilenet_int8-weights.p"
        weights_dict = pickle.load(open(weights_file,"rb"))
        weight_limits_file = "../pretrained_models/imagenet/MLperf/mobilenet_int8-weight_limits.p"
        weight_limits_dict = pickle.load(open(weight_limits_file,"rb"))
        keras_model = MobileNet_int8(weights_dict,weight_limits_dict)

    elif model_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        keras_file = "../pretrained_models/imagenet/inceptionV3.h5"
        keras_model = InceptionV3(weights=keras_file)

    ##### CIFAR-10 MODELS
    elif model_name == "ResNet56":
        keras_model = load_keras_model("../pretrained_models/cifar10/resnet56v1_cifar10.h5")
    elif model_name == "ResNet32":
        keras_model = load_keras_model("../pretrained_models/cifar10/resnet32v1_cifar10.h5")
    elif model_name == "ResNet20":
        keras_model = load_keras_model("../pretrained_models/cifar10/resnet20v1_cifar10.h5")
    elif model_name == "ResNet14":
        keras_model = load_keras_model("../pretrained_models/cifar10/resnet14v1_cifar10.h5")

    #### CIFAR-10 LARQ and WHETSTONE MODEL EXAMPLES
    elif model_name == "larq_cifar10":
        keras_model = load_keras_model("../pretrained_models/cifar10/larq/cifar10_larq.h5",custom_import=True,larq=True)
    elif model_name == "whetstone_cifar10":
        keras_model = load_keras_model("../pretrained_models/cifar10/whetstone/cifar10_whetstone.h5",custom_import=True,whetstone=True)

    ##### CIFAR-100 MODELS
    elif model_name == "ResNet56_cifar100":
        keras_model = load_keras_model("../pretrained_models/cifar100/resnet56v1_cifar100.h5")

    ##### MNIST MODELS
    elif model_name == "CNN6":
        model_path = "../pretrained_models/mnist/mnist_CNN6.h5"
        keras_model = load_keras_model(model_path)

    elif model_name == "CNN6_v2":
        model_path = "../pretrained_models/mnist/mnist_CNN6_v2.h5"
        keras_model = load_keras_model(model_path,custom_import=True)

    elif model_name == "cifar10_cnn_brelu":
        model_path = "../pretrained_models/cifar10/cifar10_cnn_brelu.h5"
        keras_model = load_keras_model(model_path,custom_import=True)

    ###### ALL OTHER UN-NAMED MODELS
    # Try to use model_name as path directly
    elif model_name is not None:
        keras_model = load_keras_model(model_name)

    else:
        raise ValueError("Model name invalid and model path not provided")

    if show_model_summary:
        keras_model.summary()

    return keras_model

'''
load_adc_activation ranges
Load calibrated ADC and activation ranges for the given hardware configuration and neural network model
    "DAC" here refers to activation ranges/resolution, even if an actual hardware DAC is not used
Calibrated ranges may not be available for the given configuration: 
    In this case, if adc_range_option is "calibrated", an error is returned.
        Either change adc_range_option, set adc_bits/dac_bits to zero, or generate calibrated limits
If adc_range_option is not "calibrated", the ranges are set to [None, None]
    The actual range used is set elsewhere in the code
Returns: calibrated ADC and activation (DAC) ranges, loaded from file
'''
def load_adc_activation_ranges(config):

    # Relevant inputs for selecting ADC and activation ranges
    model_name = config.model_name
    adc_range_option = config.adc_range_option
    adc_bits = config.adc_bits
    dac_bits = config.dac_bits
    Nlayers_mvm = config.Nlayers_mvm
    style = config.style
    NrowsMax = config.NrowsMax
    ADC_per_ibit = config.ADC_per_ibit
    Nslices = config.Nslices
    pct = config.pct

    adc_ranges = [None] * Nlayers_mvm
    dac_ranges = [None] * Nlayers_mvm

    if NrowsMax == 0:
        # An NrowsMax value of 0 really means no upper bound
        NrowsMax = 1e6

    # Ranges for the ADCs
    # This depends on several things
    #   - The model
    #   - Number of weight bit slices
    #   - The negative weight handling scheme (style)
    #   - The array size (NrowsMax)
    #   - Whether input bit accumulation is done in analog (relevant for offset and bit slicing)
    #   - For balanced core, whether current is subtracted in analog (here, always assumed true)
    # Calibrated ADC ranges are not available for all models and every configuration, but they can be generated separately

    if adc_bits > 0 and adc_range_option == "calibrated":
        # It may be possible to use these limits with NrowsMax < 1152 but significant clipping may occur!
        if model_name == "Resnet50" or model_name == "Resnet50-v1.5" or model_name == "Resnet50-int4":

            if Nslices == 1:
                if style == "BALANCED" and NrowsMax >= 1152 and not ADC_per_ibit:
                    if model_name == "Resnet50-v1.5":
                        adc_ranges = np.load("../adc/adc_limits/imagenet/ResNet50v15_ADC_limits_balanced.npy")
                    elif model_name == "Resnet50":
                        adc_ranges = np.load("../adc/adc_limits/imagenet/ResNet50_ADC_limits.npy")
                    elif model_name == "Resnet50-int4":
                        adc_ranges = np.round(np.load("../adc/adc_limits/imagenet/ResNet50int4_ADC_limits.npy"))

                elif style == "OFFSET" and NrowsMax >= 1152 and ADC_per_ibit:
                    if model_name == "Resnet50-v1.5":
                        adc_ranges = np.load("../adc/adc_limits/imagenet/ResNet50v15_ADC_limits_offset.npy")

                elif style == "BALANCED" and NrowsMax == 144 and not ADC_per_ibit:
                    if model_name == "Resnet50-v1.5":
                        adc_ranges = np.load("../adc/adc_limits/imagenet/ResNet50v15_ADC_limits_balanced_144rows.npy")

                elif style == "BALANCED" and NrowsMax == 1152 and ADC_per_ibit:
                    if model_name == "Resnet50-v1.5":
                        adc_ranges = np.load("../adc/adc_limits/imagenet/ResNet50v15_ADC_limits_balanced_ibits.npy")

            elif Nslices > 1:
                try:
                    if style == "BALANCED" and not ADC_per_ibit:
                        adc_ranges = np.load("../adc/adc_limits/imagenet_bitslicing/ResNet50v15_"+str(NrowsMax)+"rows_"+str(Nslices)+"slices_pct="+str(pct)+"_balanced.npy")
                    elif style == "OFFSET" and ADC_per_ibit:
                        adc_ranges = np.load("../adc/adc_limits/imagenet_bitslicing/ResNet50v15_"+str(NrowsMax)+"rows_"+str(Nslices)+"slices_pct="+str(pct)+"_offset.npy")
                    else:
                        raise ValueError("No calibrated ADC range found for the combination of crossbar style, ADC_per_ibit and # slices")
                except FileNotFoundError:
                    raise ValueError("No calibrated ADC range found for the combination of crossbar style, ADC_per_ibit and # slices")

        # CIFAR10 ResNet models
        elif (model_name == "ResNet56" or model_name == "ResNet32" or model_name == "ResNet20" or model_name == "ResNet14") and Nslices == 1:
            if style == "BALANCED" and not ADC_per_ibit:
                # For ResNet56, ResNet32, and ResNet20, the ADC ranges are based on NrowsMax = 576
                # Use these with caution if NrowsMax is smaller; may need new ranges
                if model_name == "ResNet56":
                    adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet56_ADC_limits.npy")
                elif model_name == "ResNet32":
                    adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet32_ADC_limits.npy")
                elif model_name == "ResNet20":
                    adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet20_ADC_limits.npy")
                elif model_name == "ResNet14":
                    if NrowsMax >= 576:
                        adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet14_ADC_limits.npy")
                    elif NrowsMax == 288:
                        adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet14_ADC_limits_288rows.npy")
                    elif NrowsMax == 144:
                        adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet14_ADC_limits_144rows.npy")
            elif style == "OFFSET" and ADC_per_ibit:
                if model_name == "ResNet14":
                    adc_ranges = np.load("../adc/adc_limits/cifar10/ResNet14_ADC_limits_offset.npy")

        elif model_name == "VGG19":
            if Nslices == 1 and style == "BALANCED" and NrowsMax >= 1152 and not ADC_per_ibit:
                adc_ranges = np.load("../adc/adc_limits/imagenet/VGG19_ADC_limits.npy")

        elif model_name == "CNN6" and Nslices == 1:
            if style == "BALANCED" and not ADC_per_ibit:
                adc_ranges = np.load("../adc/adc_limits/mnist/CNN6_ADC_limits.npy")

        elif model_name == "CNN6_v2" and Nslices == 1:
            if style == "BALANCED" and not ADC_per_ibit:
                adc_ranges = np.load("../adc/adc_limits/mnist/CNN6v2_ADC_limits.npy")

        if adc_ranges[0] is None:
            raise ValueError("For the chosen model and crossbar settings, calibrated ADC ranges are unavailable: "+\
                "please generate the limits, choose another range setting ('max' or 'granular'), or disable ADC.")

    # Ranges for the activations
    # These depend on the model but do not depend sensitively on the other parameters above
    # For ResNet50-int4, these ranges MUST be loaded, since activation quantization is part of the model
    if dac_bits > 0 or model_name == "Resnet50-int4":
        if model_name == "Resnet50-v1.5":
            dac_ranges = np.load("../adc/adc_limits/imagenet/ResNet50v15_DAC_limits.npy")
        elif model_name == "Resnet50":
            dac_ranges = np.load("../adc/adc_limits/imagenet/ResNet50_DAC_limits.npy")
        elif model_name == "Resnet50-int4":
            dac_ranges = np.load("../adc/adc_limits/imagenet/ResNet50int4_DAC_limits.npy")
        elif model_name == "ResNet56":
            dac_ranges = np.load("../adc/adc_limits/cifar10/ResNet56_DAC_limits.npy")
        elif model_name == "ResNet32":
            dac_ranges = np.load("../adc/adc_limits/cifar10/ResNet32_DAC_limits.npy")
        elif model_name == "ResNet20":
            dac_ranges = np.load("../adc/adc_limits/cifar10/ResNet20_DAC_limits.npy")
        elif model_name == "ResNet14":
            dac_ranges = np.load("../adc/adc_limits/cifar10/ResNet14_DAC_limits.npy")
        elif model_name == "CNN6":
            dac_ranges = np.load("../adc/adc_limits/mnist/CNN6_DAC_limits.npy")
        elif model_name == "CNN6_v2":
            # Bounded ReLU means DAC limits are trivial
            dac_ranges = [np.array([0,1])] * Nlayers_mvm
        elif model_name == "cifar10_cnn_brelu":
            # Bounded ReLU means DAC limits are trivial
            dac_ranges = [np.array([0,1])] * Nlayers_mvm
        elif model_name == "MobilenetV1" or model_name == "MobilenetV1-int8":
            dac_ranges = [np.array([0,6])] * Nlayers_mvm # ReLU6
            dac_ranges[0] = np.array([-1,1])
        elif model_name == "VGG19":
            dac_ranges = np.load("../adc/adc_limits/imagenet/VGG19_DAC_limits.npy")

        if dac_ranges[0] is None:
            raise ValueError("For the chosen model and crossbar settings, calibrated activation ranges are unavailable; please generate the limits or disable DAC.")

    return adc_ranges, dac_ranges


'''
Set parameters that are specific to the neural network. Edit this file when adding a new neural network
that may need special treatment.
Inputs: config object, Returns: modified config object
The following additional parameters are added by this function:
    - subtract_pixel_mean: for CIFAR-10 networks, whether the mean input values of the training set is subtracted from the test set
    - imagenet_preprocess: for ImageNet networks, chooses the preprocessing option for the dataset (None for all other networks)
    - dataset_normalization: how the loaded dataset should be normalized
        * "none": no normalization (or assumes normalization is done in saved pre-processed images)
        * "unsigned_8b": normalizes values in the range (0,255) into the range (0,1)
        * "signed_8b": normalizes values in the range (0,255) into the range (-1,1)
    - positiveInputsOnly: vector specifying whether each MVM layer has strictly positive inputs (e.g. directly follows ReLU), or not
    - memory_window: how many previous layers' activations to hold in memory during the simulation to account for skip connections
    - larq/whetstone: whether the model uses Larq or Whetstone layers
'''
def model_specific_parameters(config):

    task = config.task
    model_name = config.model_name
    Nlayers_mvm = config.Nlayers_mvm

    ########### Preprocessing parameters

    ### Subtract pixel mean
    if task == "cifar10" or task == "cifar100":
        if "ResNet" in model_name:
            subtract_pixel_mean = True
        else:
            subtract_pixel_mean = False
    else:
        subtract_pixel_mean = False

    ### ImageNet preprocess
    if task == "imagenet":
        if model_name == "Resnet50":
            imagenet_preprocess = "keras_resnet"
        elif model_name == "Resnet50-v1.5":
            imagenet_preprocess = "mlperf_resnet50"
        elif model_name == "Resnet50-int4":
            imagenet_preprocess = "torchvision"
        elif model_name == "VGG19":
            imagenet_preprocess = "keras_vgg"
        elif model_name == "MobilenetV2" or model_name == "MobilenetV1":
            imagenet_preprocess = "keras_mobilenet"
        elif model_name == "InceptionV3":
            imagenet_preprocess = "keras_inception"
        elif model_name == "MobilenetV1-int8":
            imagenet_preprocess = "mlperf_mobilenet"
    else:
        imagenet_preprocess = None

    ### Dataset normalization
    # To add new cases not listed here, add a corresponding if clause into the appropriate section
    # in inference_net.inference
    if task == "imagenet":
        dataset_normalization = "none"
    elif task == "cifar10" or task == "cifar100":
        if model_name == "larq_cifar10":
            dataset_normalization = "signed_8b"
        else:
            dataset_normalization = "unsigned_8b"
    elif task == "mnist" or task == "fashion":
        dataset_normalization = "unsigned_8b"

    # Layer-by-layer boolean of whether inputs to MVM are strictly positive due to ReLU 
    positiveInputsOnly = [False for k in range(Nlayers_mvm)]

    # Max number of previous' layers activations to save in memory, needed for non-sequential networks (e.g. ResNet)
    # Can be arbitrarily large (at the cost of memory use) but cannot be too small
    # Default value is 10 to be safe
    memory_window = 10

    # ResNet50 for ImageNet (Resnet50 = Keras version, Resnet50-v1.5 = MLPerf version)
    if model_name == "Resnet50" or model_name == "Resnet50-v1.5" or model_name == "Resnet50-int4":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 6

        # The INT4 model already assumes that activations are quantized to 4 or 7 bits
        if model_name == "Resnet50-int4":
            config.dac_bits_vec = 4 * np.ones(config.Nlayers_mvm,dtype=int)
            config.dac_bits_vec[0] = 7
            config.dac_bits_vec[-1] = 7
            # Weights are already quantized, so avoid doing unneeded quantization
            config.weight_bits = 0
            config.bias_bits = 0
            config.dac_bits = 7 # need a nonzero value for load_adc_activation_ranges
            if config.Nslices > 1:
                # TODO
                raise ValueError("This model is not currently compatible with weight bit slicing")

    # VGG19 for ImageNet (Keras version)
    elif model_name == "VGG19":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 3

    # MobileNetV2 for ImageNet (Keras version)
    elif model_name == "MobilenetV2":
        positiveInputsOnly = [False for k in range(Nlayers_mvm)]
        relu6_indices = np.array([1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,\
            37,38,40,41,43,44,46,47,49,50,52])
        for ind in relu6_indices:
            positiveInputsOnly[ind] = True
        memory_window = 6

    elif model_name == "MobilenetV1" or model_name == "MobilenetV1-int8":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 3

        if model_name == "MobilenetV1-int8":
            # Weights are already quantized to 8 bits (bias is not)
            if config.Nslices == 1:
                config.weight_bits = 0

    elif model_name == "InceptionV3":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 10

    elif model_name == "ResNet56" or model_name == "ResNet32" or model_name == "ResNet20" or model_name == "ResNet14" \
        or model_name == "ResNet56_cifar100":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 3

    elif model_name == "CNN6" or model_name == "CNN6_v2":
        # Lumping all MNIST CNN models together
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        memory_window = 1

    elif model_name == "cifar10_cnn_brelu":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        memory_window = 1

    elif model_name == "larq_cifar10":
        positiveInputsOnly = [False for k in range(Nlayers_mvm)]
        memory_window = 6

    elif model_name == "whetstone_cifar10":
        positiveInputsOnly = [True for k in range(Nlayers_mvm)]
        positiveInputsOnly[0] = False
        memory_window = 6

    else:
        # Use default values
        positiveInputsOnly = [False for k in range(Nlayers_mvm)]
        print("Assuming signed inputs to every layer")

    # Special layer types
    larq, whetstone = False, False
    if model_name == "larq_cifar10":
        larq = True
    if model_name == "whetstone_cifar10":
        whetstone = True

    # Set model-specific config parameters
    config.imagenet_preprocess = imagenet_preprocess
    config.subtract_pixel_mean = subtract_pixel_mean
    config.dataset_normalization = dataset_normalization
    config.memory_window = memory_window
    config.larq = larq
    config.whetstone = whetstone

    return config, positiveInputsOnly

'''
get_xy_parallel:
This function sets the number of sliding windows in x and y to pack into a single MVM simulation
* The default case (disable) is (1,1) for all layers. This guarantees that the simulation
    uses the minimum amount of memory, but may be slow
* For optimal results, these numbers should be separately tuned for:
    - Different neural network models
    - Different layers of the neural network
    - Different hardware settings, especially: # weight bit slices, array size, crossbar style, read noise
        If Rp > 0, see get_xy_parallel_parasitics()
    - GPU vs CPU
* Some numbers have been provided below for selected models/hardware settings.
    Note: THESE VALUES HAVE BEEN OPTIMIZED FOR A GPU WITH 32GB OF MEMORY
    Setting these values too large will result in OutOfMemory errors. Reduce the values accordingly
    In these examples, the same sliding window packing was applied to groups of layers. The optimal
        grouping was found empricially
Returns: (N, 2) array where N = # MVM layers. Each tuple corresponds to the number of sliding windows in (x, y)
    to run in parallel. If the MVM layer is not a convolution, it is ignored
'''
def get_xy_parallel(config, disable=False):

    model_name = config.model_name
    Nlayers_mvm = config.Nlayers_mvm
    fast_balanced = config.fast_balanced
    Nslices = config.Nslices
    style = config.style

    # Default: No parallelization of sliding windows
    xy_pars = np.ones((Nlayers_mvm,2),dtype=int)
    if disable:
        return xy_pars

    ## In general, the limits abelow assume Nslices = 1
    # Adjust according to GPU memory availability for Nslices > 1

    # ResNet50 for ImageNet (Resnet50 = Keras version, Resnet50-v1.5 = MLPerf version)
    if model_name == "Resnet50" or model_name == "Resnet50-v1.5" or model_name == "Resnet50-int4":

        single_array_slice = (fast_balanced or style == "OFFSET")

        if Nslices == 1 and single_array_slice:
            k_chkpt = [0,8,20,43]
            for k in range(Nlayers_mvm):
                if k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([14,14],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([14,7],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([4,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,7],dtype=int)

        elif Nslices == 1 or (Nslices == 2 and single_array_slice):
            k_chkpt = [0,8,11,20]
            for k in range(Nlayers_mvm):
                if k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([14,14],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([8,7],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([4,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,7],dtype=int)

        elif Nslices == 2 or (Nslices == 4 and single_array_slice):
            k_chkpt = [0,8,11,20,45]
            for k in range(Nlayers_mvm):
                if k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([14,14],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([4,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[4]:
                   xy_pars[k,:] = np.array([1,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,1],dtype=int)

        elif Nslices == 4 or (Nslices == 8 and single_array_slice):
            k_chkpt = [0,8,11,20,43]
            for k in range(Nlayers_mvm):
                if k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([4,7],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[4]:
                   xy_pars[k,:] = np.array([1,1],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,1],dtype=int)

    # VGG19 for ImageNet (Keras version)
    elif model_name == "VGG19":
        for k in range(Nlayers_mvm):
            k_chkpt = [1,3,7,11,15]
            for k in range(Nlayers_mvm):
                if k == 0:
                   xy_pars[k,:] = np.array([14,14],dtype=int)
                elif k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([2,7],dtype=int)
                elif k <= k_chkpt[4]:
                   xy_pars[k,:] = np.array([1,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,1],dtype=int)

    elif model_name == "MobilenetV2":
        for k in range(Nlayers_mvm):
            k_chkpt = [3,9,18,39]
            for k in range(Nlayers_mvm):
                if k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([4,4],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([4,4],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([1,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,7],dtype=int)

    elif model_name == "MobilenetV1" or model_name == "MobilenetV1-int8":
        for k in range(Nlayers_mvm):
            k_chkpt = [2,6,10,22]
            for k in range(Nlayers_mvm):
                if k == 0:
                   xy_pars[k,:] = np.array([16,16],dtype=int)
                elif k <= k_chkpt[0]:
                   xy_pars[k,:] = np.array([16,16],dtype=int)
                elif k <= k_chkpt[1]:
                   xy_pars[k,:] = np.array([8,8],dtype=int)
                elif k <= k_chkpt[2]:
                   xy_pars[k,:] = np.array([4,7],dtype=int)
                elif k <= k_chkpt[3]:
                   xy_pars[k,:] = np.array([1,7],dtype=int)
                else:
                   xy_pars[k,:] = np.array([1,7],dtype=int)
            # Last conv is really a dense layer
            xy_pars[-1,:] = np.array([1,1],dtype=int)

    elif model_name == "InceptionV3":
        for k in range(Nlayers_mvm):
            k_chkpt = [2,3,4,27,73]
            for k in range(Nlayers_mvm):
                if k == 0:
                   xy_pars[k,:] = np.array([1,149],dtype=int)
                elif k <= k_chkpt[0]: # 147
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[1]: # 73 -- prime!
                   xy_pars[k,:] = np.array([1,73],dtype=int)
                elif k <= k_chkpt[2]: # 71 -- prime!
                   xy_pars[k,:] = np.array([1,71],dtype=int)
                elif k <= k_chkpt[3]: # 35
                   xy_pars[k,:] = np.array([7,7],dtype=int)
                elif k <= k_chkpt[4]: # 17
                   xy_pars[k,:] = np.array([1,17],dtype=int)
                else: # 
                   xy_pars[k,:] = np.array([2,2],dtype=int)

    elif model_name == "ResNet56":
        xy_pars = np.zeros((57,2),dtype=int)
        for k in range(57):
            if k == 0:
                xy_pars[k,:] = np.array([32,32],dtype=int)
            elif k ==1 or k == 19:
                xy_pars[k,:] = np.array([16,16],dtype=int)
            elif k <= 18 or k == 21:
                xy_pars[k,:] = np.array([16,8],dtype=int)
            elif k <= 37:
                xy_pars[k,:] = np.array([8,4],dtype=int)
            else:
                xy_pars[k,:] = np.array([4,4],dtype=int)

    elif model_name == "ResNet32":
        xy_pars = np.zeros((33,2),dtype=int)
        for k in range(33):
            if k == 0:
                xy_pars[k,:] = np.array([32,32],dtype=int)
            elif k <= 12:
                xy_pars[k,:] = np.array([16,8],dtype=int)
            elif k <= 21:
                xy_pars[k,:] = np.array([8,8],dtype=int)
            else:
                xy_pars[k,:] = np.array([4,4],dtype=int)

    elif model_name == "ResNet20":
        xy_pars = np.zeros((21,2),dtype=int)
        for k in range(21):
            if k == 0:
                xy_pars[k,:] = np.array([32,32],dtype=int)
            elif k <= 6:
                xy_pars[k,:] = np.array([16,16],dtype=int)
            elif k <= 13:
                xy_pars[k,:] = np.array([8,8],dtype=int)
            else:
                xy_pars[k,:] = np.array([8,4],dtype=int)

    elif model_name == "ResNet14":
        xy_pars = np.zeros((15,2),dtype=int)
        for k in range(15):
            if k == 0:
                xy_pars[k,:] = np.array([32,32],dtype=int)
            elif k <= 4:
                xy_pars[k,:] = np.array([16,16],dtype=int)
            elif k <= 9:
                xy_pars[k,:] = np.array([16,8],dtype=int)
            else:
                xy_pars[k,:] = np.array([4,4],dtype=int)

    elif model_name == "ResNet56_cifar100":
        for k in range(57):
            if k == 0:
                xy_pars[k,:] = np.array([16,16],dtype=int)
            elif k ==1 or k == 19:
                xy_pars[k,:] = np.array([8,8],dtype=int)
            elif k <= 18 or k == 21:
                xy_pars[k,:] = np.array([4,4],dtype=int)
            elif k <= 37:
                xy_pars[k,:] = np.array([4,4],dtype=int)
            else:
                xy_pars[k,:] = np.array([2,2],dtype=int)

    elif model_name == "CNN6" or model_name == "CNN6_v2":
        xy_pars = np.ones((4,2),dtype=int)
        xy_pars[0,:] = np.array([28,28])
        xy_pars[1,:] = np.array([28,28])
        xy_pars[2,:] = np.array([14,14])
        xy_pars[3,:] = np.array([14,14])

    elif model_name == "cifar10_cnn_brelu":
        xy_pars = np.ones((4,2),dtype=int)
        xy_pars[0,:] = np.array([32,16])
        xy_pars[1,:] = np.array([16,8])
        xy_pars[2,:] = np.array([4,4])
        xy_pars[3,:] = np.array([4,4])

    elif model_name == "larq_cifar10":
        k_chkpt = [3,5]
        for k in range(Nlayers_mvm):
            if k <= k_chkpt[0]:
               xy_pars[k,:] = np.array([5,5],dtype=int)
            elif k <= k_chkpt[1]:
               xy_pars[k,:] = np.array([7,1],dtype=int)
            else:
               xy_pars[k,:] = np.array([3,3],dtype=int)

    elif model_name == "whetstone_cifar10":
        for k in range(Nlayers_mvm):
           xy_pars[k,:] = np.array([2,2],dtype=int)

    return xy_pars

'''
get_xy_parallel_parasitics:
Same as get_xy_parallel(), but specifically for the case where parasitic resistance is enabled (Rp > 0)
'''
def get_xy_parallel_parasitics(Nrows,Nx_in,Nx_out,model_name):

    # Default xy_par if model is not in the list below
    xy_par = np.array([1,1])

    if model_name == "Resnet50-v1.5" or model_name == "Resnet50" or model_name == "Resnet50-int4":
        # Layers that can use an even xy_par value
        if Nx_out != 7:
            if Nrows in (64,128,147):
                xy_par = np.array([4,4])
            elif Nrows == 256:
                if Nx_in > 14:
                    xy_par = np.array([4,2])
                else:
                    xy_par = np.array([7,1])
            elif Nrows == 512:
                xy_par = np.array([7,1])
            elif Nrows == 576:
                xy_par = np.array([2,2])
            elif Nrows in (896,1024):
                xy_par = np.array([2,1])
            elif Nrows >= 1152:
                xy_par = np.array([2,1])
            else:
                raise ValueError("Need to specify parasitic x_par, y_par for array size: "+str(Nrows))
        # Layers that can only use 1 or 7 as xy_par value
        else:
            if Nrows == 512:
                xy_par = np.array([7,1])
            else:
                xy_par = np.array([1,1])

    elif model_name == "ResNet56" or model_name == "ResNet32" or model_name == "ResNet20" or model_name == "ResNet14":
        # Nrows = 16, 27, 32, 64, 144, 288, 576
        if Nrows <= 32:
            xy_par = np.array([8,4])
        elif Nrows == 64:
            xy_par = np.array([4,4])
        elif Nrows == 144:
            xy_par = np.array([4,4])
        elif Nrows == 288:
            xy_par = np.array([4,2])
        elif Nrows == 576:
            xy_par = np.array([2,2])
        else:
            raise ValueError("Need to specify parasitic x_par, y_par for array size: "+str(Nrows))

    return xy_par