#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
sys.path.append("..")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
This function imports the neural network layers metadata from Keras
The weights themselves are imported later, in BackProp
"""    
def get_keras_metadata(model,debug_graph=False,task="imagenet"):
    print('Reading Keras model metadata...')
    config = model.get_config()['layers']

    ignoredLayerTypes = ["Dropout","GaussianNoise"]

    # The main layer types are: Conv, Dense, Pool, Add, Quantize, Concat, and Scale
    #   Each layer can have an associated activation layer: 'activation' (dict, 'name')
    #   Each layer can have an associated batch normalization layer: 'batch_norm'
    #       Currently not supporting batch norm layers that are not associated with any single Conv or Dense layer
    #   Any additional appended layer is part of 'appended'
    #       These include: GaussianNoise, ZeroPadding2D, and Round, which are seldom used
    layerParams = []

    def isActivation(class_name):
        return class_name in ("Activation", "ReLU", "Spiking_BRelu", "Softmax_Decode")

    def isMainLayerType(class_name):
        return class_name in ('Conv2D', 'DepthwiseConv2D', 'QuantConv2D','MaxPooling2D','AveragePooling2D',\
            'GlobalAveragePooling2D','GlobalMaxPooling2D','Dense','QuantDense','Add','Concatenate','Quantize','Scale')

    # Search the network for the index of a named main layer or the main layer associated with a named supporting layer
    def searchForLayer(layerName,layerParams):
        for j in range(len(layerParams)):
            if (layerParams[j]['name'] == layerName) \
            or (layerParams[j]['batch_norm'] == layerName) or (layerParams[j]['appended'] == layerName) \
            or (layerParams[j]['activation'] is not None and layerParams[j]['activation']['name'] == layerName):
                return j
        raise ValueError("Source layer could not be found")

    # Iterate through the layers of the network
    for k in range(len(config)):
        class_name = config[k]['class_name']
        config_k = config[k]['config']

        # Input layer of network: specifies input dimensions of the first layer
        # Not always specified in a Keras CNN model; in that case use the dataset image size as a guess
        if class_name == 'InputLayer':
            Nix0 = config_k['batch_input_shape'][1]
            if len(config_k['batch_input_shape']) == 4: # 3D input
                Niy0 = config_k['batch_input_shape'][2]
                Nic0 = config_k['batch_input_shape'][3]
                if Nix0 is None or Niy0 is None:
                    print('Input dimensions not defined. Using '+task+' image size')
                    if task in ("cifar10","cifar100"):
                        Nix0, Niy0 = 32, 32
                    elif task in ("mnist","fashion"):
                        Nix0, Niy0 = 28, 28
                    elif task == "tinyimagenet":
                        Nix0, Niy0 = 64, 64
                    elif task == "imagenet":
                        Nix0, Niy0 = 224, 224
                    else:
                        raise ValueError("Image size not specified and dataset unknown")

        # Zero padding layer is detected, but its parameters are loaded upon parsing subsequent Conv layer
        elif class_name == 'ZeroPadding2D':
            if len(layerParams) > 0:
                input1 = config[k]['inbound_nodes'][0][0][0]
                k_src = searchForLayer(input1,layerParams)
                layerParams[k_src]['appended'] = config_k['name']

        # Check if layer is a main layer type
        elif isMainLayerType(class_name):

            layerParams_k = {}
            layerParams_k['name'] = config_k['name']
            
            # Default settings: may change below
            layerParams_k['batch_norm'] = None
            layerParams_k['appended'] = None
            layerParams_k['bias'] = False
            layerParams_k['activation'] = None
            layerParams_k['splitBeforeBN'] = False

            ###################
            ##  CONVOLUTION  ##
            ###################
            if class_name in ('Conv2D', 'DepthwiseConv2D', 'QuantConv2D'):

                layerParams_k['type'] = 'conv'
                layerParams_k['bias'] = config_k['use_bias']
                layerParams_k['stride'] = config_k['strides'][0] # separate stride in y not yet supported
                layerParams_k['Kx'] = config_k['kernel_size'][0]
                layerParams_k['Ky'] = config_k['kernel_size'][1]
                layerParams_k['binarizeWeights'] = False # used only for Larq, changed below if detected

                # Zero padding
                # Default to zero padding: if same convolution, compute these later in post_set
                layerParams_k['px_0'], layerParams_k['px_1'] = 0, 0
                layerParams_k['py_0'], layerParams_k['py_1'] = 0, 0
                if config_k['padding'] == 'same':
                    layerParams_k['sameConv'] = True
                elif config_k['padding'] == 'valid':
                    layerParams_k['sameConv'] = False
                    if config[k-1]['class_name'] == 'ZeroPadding2D':
                        layerParams_k['px_0'] = config[k-1]['config']['padding'][1][0]
                        layerParams_k['px_1'] = config[k-1]['config']['padding'][1][1]
                        layerParams_k['py_0'] = config[k-1]['config']['padding'][0][0]
                        layerParams_k['py_1'] = config[k-1]['config']['padding'][0][1]

                # Check if convolution is depthwise and set # output channels (set later for depthwise)
                layerParams_k['depthwise'] = (class_name == 'DepthwiseConv2D')
                if not layerParams_k['depthwise']:
                    layerParams_k['Noc'] = config_k['filters']

                # Input shape can be specified for the first conv layer, computed later for the other layers
                if k == 0 and len(layerParams) == 0: # No InputLayer
                    layerParams_k['Nix'] = config_k['batch_input_shape'][1]
                    layerParams_k['Niy'] = config_k['batch_input_shape'][2]
                    layerParams_k['Nic'] = config_k['batch_input_shape'][3]
                elif len(layerParams) == 0: # Get dimensions from InputLayer
                    layerParams_k['Nix'] = Nix0
                    layerParams_k['Niy'] = Niy0
                    layerParams_k['Nic'] = Nic0

                # Find source layer, if no inbound nodes set, assume sequential
                if len(layerParams) > 0:
                    if 'inbound_nodes' in config[k]:
                        input1 = config[k]['inbound_nodes'][0][0][0]
                        k_src = searchForLayer(input1,layerParams)
                    else:                        
                        k_src = len(layerParams)-1
                    layerParams_k['source'] = np.array([k_src])
                    layerParams_k['splitBeforeBN'] = ( (layerParams[k_src]['type'] == 'add') and ('add' in input1) )
                else:
                    layerParams_k['source'] = None # First layer
                
                # Check if activation is defined within the conv layer
                if 'activation' in config_k and config_k['activation'] is not None and config_k['activation'] != "linear":
                    if config_k['activation'] == 'relu':
                        activation = {}
                        activation['name'] = config_k['name']+'_relu'
                        activation['type'] = "RECTLINEAR"
                        activation['bound'] = 1e20
                        layerParams_k['activation'] = activation
                    else:
                        raise ValueError("Unrecognized activation in conv layer")

            ###################
            ##    POOLING    ##
            ###################
            elif class_name in ('MaxPooling2D','AveragePooling2D','GlobalAveragePooling2D','GlobalMaxPooling2D'):
                
                # If a max pool layer is found, modify the parameters of the last conv layer
                layerParams_k['type'] = 'pool'
                if class_name in ('MaxPooling2D','GlobalMaxPooling2D'):
                    layerParams_k['poolType'] = 'max'
                elif class_name in ('AveragePooling2D','GlobalAveragePooling2D'):
                    layerParams_k['poolType'] = 'avg'

                # Pooling kernel size: set later for Global pools
                if class_name not in ('GlobalAveragePooling2D','GlobalMaxPooling2D'):
                    layerParams_k['MPx'] = config_k['pool_size'][0]
                    layerParams_k['MPy'] = config_k['pool_size'][1]
                    layerParams_k['stride_MP'] = config_k['strides'][0] # separate stride in y not yet supported
                else:
                    layerParams_k['MPx'] = 0 # placeholder for later computation
                    layerParams_k['MPy'] = 0
                    layerParams_k['stride_MP'] = 1

                # Padding: determine the type of padding here, compute the padding later in the function
                if 'padding' in config_k:
                    layerParams_k['padding'] = config_k['padding']
                else:
                    layerParams_k['padding'] = None

                # If this layer comes after an explicit padding layer, add it now
                if config[k-1]['class_name'] == 'ZeroPadding2D':
                    layerParams_k['py_L'] = config[k-1]['config']['padding'][0][0]
                    layerParams_k['py_R'] = config[k-1]['config']['padding'][0][1]
                    layerParams_k['px_L'] = config[k-1]['config']['padding'][1][0] 
                    layerParams_k['px_R'] = config[k-1]['config']['padding'][1][1]
                else:
                    # These are for valid-type padding. If same-type padding, will be recomputed layer                
                    layerParams_k['px_L'], layerParams_k['px_R'] = 0, 0
                    layerParams_k['py_L'], layerParams_k['py_R'] = 0, 0

                # Set avgpool rounding option    
                layerParams_k['round'] = (config[k+1]['class_name'] == 'Round')

                # Find source layer
                if 'inbound_nodes' in config[k]:
                    input1 = config[k]['inbound_nodes'][0][0][0]
                    k_src = searchForLayer(input1,layerParams)
                else:
                    k_src = len(layerParams)-1
                layerParams_k['source'] = np.array([k_src])
                layerParams_k['splitBeforeBN'] = ( (layerParams[k_src]['type'] == 'add') and ('add' in input1) )   

            ###################
            ##     DENSE     ##
            ###################
            elif class_name in ('Dense','QuantDense'):
                layerParams_k['type'] = 'dense'
                layerParams_k['units'] = config_k['units'] # number of outputs
                layerParams_k['bias'] = config_k['use_bias']
                layerParams_k['binarizeWeights'] = False # used only for Larq

                # Find source layer
                # If layer is preceded by flatten, find the source layer of flatten
                if len(layerParams) > 0:
                    if 'inbound_nodes' in config[k]:
                        input1 = config[k]['inbound_nodes'][0][0][0]
                        if "flatten" in input1:
                            for q in range(len(config)):
                                if config[q]['config']['name'] == input1:
                                    input1 = config[q]['inbound_nodes'][0][0][0]
                                    break
                        k_src = searchForLayer(input1,layerParams)
                    else:
                        k_src = len(layerParams)-1
                    layerParams_k['source'] = np.array([k_src])
                    layerParams_k['splitBeforeBN'] = ( (layerParams[k_src]['type'] == 'add') and ('add' in input1) )
                else:
                    layerParams_k['source'] = None # Dense layer is first layer
                
                # Check if activation is defined within the conv layer
                if 'activation' in config_k and config_k['activation'] is not None and config_k['activation'] != "linear":
                    activation = {}
                    if config_k['activation'] == 'relu':
                        activation['name'] = config_k['name']+'_relu'
                        activation['type'] = "RECTLINEAR"
                        activation['bound'] = 1e20
                    elif config_k['activation'] == 'softmax':
                        activation['name'] = config_k['name']+'_softmax'
                        activation['type'] = "SOFTMAX"
                    else:
                        raise ValueError("Unrecognized activation in dense layer")
                    layerParams_k['activation'] = activation             

            #########################
            ##  ADD & CONCATENATE  ##
            #########################
            elif class_name in ('Add','Concatenate'):

                # Find source layers
                if 'inbound_nodes' in config[k]:
                    Nsources = len(config[k]['inbound_nodes'][0])
                    k_srcs = np.zeros(Nsources,dtype=int)
                    for q in range(Nsources):
                        input_q = config[k]['inbound_nodes'][0][q][0]
                        k_srcs[q] = searchForLayer(input_q,layerParams)
                        # input1 = config[k]['inbound_nodes'][0][0][0]
                        # input2 = config[k]['inbound_nodes'][0][1][0]
                        # k_src1 = searchForLayer(input1,layerParams)
                        # k_src2 = searchForLayer(input2,layerParams)
                else:
                    raise ValueError("inbound_nodes property must exist for Add/Concatenate layer")
                # layerParams_k['source'] = np.array([k_src1,k_src2])
                layerParams_k['source'] = k_srcs

                if class_name == 'Add':
                    layerParams_k['type'] = 'add'
                    if Nsources == 2:
                        input1 = config[k]['inbound_nodes'][0][0][0]
                        input2 = config[k]['inbound_nodes'][0][1][0]
                        if layerParams[k_srcs[0]]['type'] == 'add':
                            layerParams_k['splitBeforeBN'] = ('add' in input1)
                        elif layerParams[k_srcs[1]]['type'] == 'add':
                            layerParams_k['splitBeforeBN'] = ('add' in input2)
                elif class_name == 'Concatenate':
                    layerParams_k['type'] = 'concat'                
                    if config_k['axis'] != -1 and config_k['axis'] != 3:
                        raise ValueError("Contatenation only supported along channel dimension")
            
            #################################
            ##  NVIDIA INT4 CUSTOM LAYERS  ##
            #################################
            elif class_name in ('Quantize','Scale'):
                if class_name == 'Quantize':
                    # Custom quantization/de-quantization layer used in Nvidia's int4 network
                    layerParams_k['type'] = 'quantize'
                    layerParams_k['shift_bits'] = config_k['shift_bits']
                    layerParams_k['output_bits'] = config_k['output_bits']
                    layerParams_k['signed'] = config_k['signed']
                elif class_name == 'Scale':
                    # Layer that is used to do channel-wise division by a weight vector
                    layerParams_k['type'] = 'scale'
                    layerParams_k['units'] = config_k['units']
                
                # Name of the input layer
                if 'inbound_nodes' in config[k]:
                    input1 = config[k]['inbound_nodes'][0][0][0]
                    k_src = searchForLayer(input1,layerParams)
                else:
                    k_src = len(layerParams)-1
                layerParams_k['source'] = np.array([k_src])

            ##########################
            ##  LARQ CUSTOM LAYERS  ##
            ##########################
            if class_name in ('QuantConv2D','QuantDense'):
                
                # Weight quantization setting
                if config_k['kernel_quantizer'] is not None:
                    if config_k['kernel_quantizer']['class_name'] == 'SteSign':
                        layerParams_k['binarizeWeights'] = True
                    else:
                        raise ValueError("Kernel quantizer not recognized")
                # Change the activation of the previous layer
                if config_k['input_quantizer'] is not None:
                    if config_k['input_quantizer']['class_name'] == 'SteSign':
                        activation = {}
                        activation['name'] = layerParams[k_src]['name']+'_sign'
                        activation['type'] = "SIGN"
                        layerParams[k_src]['activation'] = activation
                    else:
                        raise ValueError("Input quantizer not recognized")

            # Add layer properties to list
            layerParams.append(layerParams_k)

        ############################################################
        ##  BATCH NORMALIZATION, ACTIVATION, and APPENDED LAYERS  ##
        ############################################################
        elif class_name in ('BatchNormalization','Round','Reshape') or class_name in ignoredLayerTypes or isActivation(class_name):

            # Find source layer
            if 'inbound_nodes' in config[k]:
                input1 = config[k]['inbound_nodes'][0][0][0]
                k_src = searchForLayer(input1,layerParams)
            else:
                k_src = len(layerParams)-1

            # Reshape is only implemented in the trivial case where it is equivalent to np.squeeze()
            if class_name == "Reshape":
                target_shape = np.array(config_k['target_shape'])
                if np.sum(target_shape>1) > 1:
                    raise ValueError("Reshape layer is unimplemented, except in the trivial case")

            # Batchnorm: associate parameters with previous layer
            if class_name == 'BatchNormalization':
                layerParams[k_src]['batch_norm'] = config_k['name']
                layerParams[k_src]['epsilon'] = config_k['epsilon']
                layerParams[k_src]['BN_scale'] = config_k['scale']
                layerParams[k_src]['BN_center'] = config_k['center']            

            elif (class_name in ('Round','Reshape') or class_name in ignoredLayerTypes) and len(layerParams) > 0:
                layerParams[k_src]['appended'] = config_k['name']

            # Bind the activation to the relevant layer
            elif isActivation(class_name):
                activation = {}
                activation['name'] = config_k['name']

                # Activation class: ReLU or sigmoid
                if config[k]['class_name'] == 'Activation':
                    if config[k]['config']['activation'] == 'relu':
                        activation['type'] = "RECTLINEAR"
                        activation['bound'] = 1e20
                    elif config[k]['config']['activation'] == 'sigmoid':
                        activation['type'] = "SIGMOID"
                    elif config[k]['config']['activation'] == 'softmax':
                        activation['type'] = "SOFTMAX"
                    elif config[k]['config']['activation'] == 'quantized_relu':
                        activation['type'] = "QUANTIZED_RELU"
                        activation['nbits'] = 8
                    else:
                        raise ValueError("Unrecognized activation function in layer type: Activation")
                # ReLU class: ReLU or bounded ReLU
                elif config[k]['class_name'] == 'ReLU':
                    activation['type'] = "RECTLINEAR"
                    if config[k]['config']['max_value'] is None:
                        activation['bound'] = 1e20
                    else:
                        activation['bound'] = config[k]['config']['max_value']*1.0
                # Spiking ReLU
                elif config[k]['class_name'] == 'Spiking_BRelu':
                    activation['type'] = "WHETSTONE"
                    activation['sharpness'] = config[k]['config']['sharpness']
                    sharpness = activation['sharpness']
                # Softmax whetstone
                elif config[k]['class_name'] == 'Softmax_Decode':
                    activation['type'] = "WHETSTONE"
                    activation['sharpness'] = sharpness

                layerParams[k_src]['activation'] = activation

        elif class_name != "Flatten":
            raise ValueError("Unrecognized Keras layer type "+class_name)

    #################################
    ##  COMPUTE FEATURE MAP SIZES  ##
    #################################
    # i^th entry is the input feature map size of i^th main layer, final entry is output feature map size of network
    sizes = [None for i in range(len(layerParams)+1)]
    for j in range(len(layerParams)):

        if j > 0:
            j_src = layerParams[j]['source'][0]

        if layerParams[j]['type'] == 'conv':
            # Caclulate input fmap size for a layer that is not the first in the network
            if j != 0:
                layerParams[j]['Nix'] = layerParams[j_src]['Nox']
                layerParams[j]['Niy'] = layerParams[j_src]['Noy']
                layerParams[j]['Nic'] = layerParams[j_src]['Noc']
                
                if layerParams[j]['depthwise']:
                    layerParams[j]['Noc'] = layerParams[j]['Nic']

            sizes[j] = (layerParams[j]['Nix'], layerParams[j]['Niy'], layerParams[j]['Nic'])

            # Compute output feature map size
            if layerParams[j]['sameConv']:
                layerParams[j]['Nox'] = layerParams[j]['Nix']//layerParams[j]['stride']
                layerParams[j]['Noy'] = layerParams[j]['Niy']//layerParams[j]['stride']
            else:
                layerParams[j]['Nox'] = 1 + (layerParams[j]['Nix']-layerParams[j]['Kx']+layerParams[j]['px_0']+layerParams[j]['px_1'])//layerParams[j]['stride']
                layerParams[j]['Noy'] = 1 + (layerParams[j]['Niy']-layerParams[j]['Ky']+layerParams[j]['py_0']+layerParams[j]['py_1'])//layerParams[j]['stride']

            # Occasionally, conv is the final layer
            if j == len(layerParams)-1:
                sizes[j+1] = (layerParams[j]['Nox'], layerParams[j]['Noy'], layerParams[j]['Noc'])

        elif layerParams[j]['type'] == 'add':
            Nsources = len(layerParams[j]['source'])
            size0 = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
            for q in range(1,Nsources):
                j_src_q = layerParams[j]['source'][q]
                size_q = (layerParams[j_src_q]['Nox'], layerParams[j_src_q]['Noy'], layerParams[j_src_q]['Noc'])
                if size0 != size_q: raise ValueError("Incoming feature map dimensions to Add layer do not match")
            sizes[j] = size0
            layerParams[j]['Nox'] = layerParams[j_src]['Nox']
            layerParams[j]['Noy'] = layerParams[j_src]['Noy']
            layerParams[j]['Noc'] = layerParams[j_src]['Noc']

        elif layerParams[j]['type'] == 'concat':
            Nsources = len(layerParams[j]['source'])
            size0 = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
            Noc_out = size0[2]
            for q in range(1,Nsources):
                j_src_q = layerParams[j]['source'][q]
                size_q = (layerParams[j_src_q]['Nox'], layerParams[j_src_q]['Noy'], layerParams[j_src_q]['Noc'])
                if (size0[0] != size_q[0]) or (size0[1] != size_q[1]):
                    raise ValueError("Incoming feature map dimensions to Concat layer do not match")
                Noc_out += size_q[2]
            sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], Noc_out)
            layerParams[j]['Nox'] = layerParams[j_src]['Nox']
            layerParams[j]['Noy'] = layerParams[j_src]['Noy']
            layerParams[j]['Noc'] = Noc_out

        elif layerParams[j]['type'] == 'pool':
            sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
            layerParams[j]['Noc'] =  layerParams[j_src]['Noc']

            # Detect if pooling type is global
            if layerParams[j]['MPx'] == 0 or layerParams[j]['MPy'] == 0:
                layerParams[j]['MPx'] = sizes[j][0]
                layerParams[j]['MPy'] = sizes[j][1]
            
            # Handle same padding in pooling layers
            # The code below is functionally identical to the padding logic in convolution_parameters.py
            MPx = layerParams[j]['MPx']
            MPy = layerParams[j]['MPy']
            Nix = layerParams[j_src]['Nox']
            Niy = layerParams[j_src]['Noy']
            stride = layerParams[j]['stride_MP']
            if layerParams[j]['padding'] == 'same':
                layerParams[j]['Nox'] = Nix // stride
                layerParams[j]['Noy'] = Niy // stride
                if (MPx % 2 != 0) and (MPy % 2 != 0):
                    # Odd size filter
                    if (Nix % stride == 0):
                        px = max(MPx - stride, 0)
                    else:
                        px = max(MPx - (Nix % stride), 0)
                    if (Niy % stride == 0):
                        py = max(MPy - stride, 0)
                    else:
                        py = max(Ky - (Niy % stride), 0)
                else:
                    # Even size filter
                    px = (layerParams[j]['Nox'] - 1)*stride + MPx - Nix
                    py = (layerParams[j]['Noy'] - 1)*stride + MPy - Niy
                layerParams[j]['px_L'] = px // 2
                layerParams[j]['px_R'] = px - layerParams[j]['px_L']
                layerParams[j]['py_L'] = py // 2
                layerParams[j]['py_R'] = py - layerParams[j]['py_L']

            else:
                # This is used for valid padding and ZeroPadding2D
                layerParams[j]['Nox'] = 1 + (Nix-MPx+layerParams[j]['px_L']+layerParams[j]['px_R']) // stride
                layerParams[j]['Noy'] = 1 + (Niy-MPy+layerParams[j]['py_L']+layerParams[j]['py_R']) // stride

            if j == len(layerParams)-1:
                sizes[j+1] = (layerParams[j]['Nox'], layerParams[j]['Noy'], layerParams[j]['Noc'])
            
        elif layerParams[j]['type'] == 'dense':
            if j != 0:
                if layerParams[j_src]['type'] in ("conv","pool","add"):
                    sizes[j] = (1,1,layerParams[j_src]['Nox']*layerParams[j_src]['Noy']*layerParams[j_src]['Noc'])
                else:
                    sizes[j] = (1,1,layerParams[j_src]['units'])
            else:
                sizes[j] = (1,1,Nix0) # Dense is first layer: input must be 1D

            if j == len(layerParams)-1:
                sizes[j+1] = (1,1,layerParams[j]['units'])

        elif layerParams[j]['type'] == 'quantize':
            # This assumes no quantization layers are present after flattening
            sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
            layerParams[j]['Nox'] = layerParams[j_src]['Nox']
            layerParams[j]['Noy'] = layerParams[j_src]['Noy']
            layerParams[j]['Noc'] = layerParams[j_src]['Noc']

        elif layerParams[j]['type'] == 'scale':
            # This assumes no scale layers are present before flattening
            sizes[j] = (1,1,layerParams[j_src]['units'])
            if j == len(layerParams)-1:
                sizes[j+1] = (1,1,layerParams[j]['units'])

        # For debug
        if debug_graph:
            print('Active layer: '+layerParams[j]['name'])
            if layerParams[j]['type'] == 'conv':
                print('     Kernel: '+str(layerParams[j]['Kx'])+' x '+str(layerParams[j]['Ky']))
                print('     Channels: '+str(layerParams[j]['Noc'])+' x '+str(layerParams[j]['Noc']))
                print('     Strides:'+str(layerParams[j]['stride'])+' x '+str(layerParams[j]['stride']))
            if layerParams[j]['source'] is not None:
                print('   Source layer 1: '+layerParams[layerParams[j]['source'][0]]['name'])
                if len(layerParams[j]['source']) > 1:
                    print('   Source layer 2: '+layerParams[layerParams[j]['source'][1]]['name'])
                print('   Take pure add as input: '+str(layerParams[j]['splitBeforeBN']))
            if layerParams[j]['batch_norm'] is not None:
                print('   Batchnorm layer: '+layerParams[j]['batch_norm'])
            else:
                print('   Batchnorm layer: None')
            if layerParams[j]['activation'] is not None:
                print('   Activation layer: '+layerParams[j]['activation']['type'])
            else:
                print('   Activation layer: None')

    # Print sizes for the MVM layers
    if debug_graph:
        i_mvm = 0
        for i in range(len(layerParams)):
            if (layerParams[i]['type'] in ('conv','dense')):
                print(layerParams[i]['name'])
                print(str(i_mvm) + ': ('+str(sizes[i+1][0])+', '+str(sizes[i+1][1])+')')
                i_mvm += 1


    if debug_graph:
        input('Keras parser debug_graph: Press any key to continue')

    return layerParams, sizes

"""
This function imports the neural network layers metadata from Keras
The weights themselves are imported later, in BackProp
"""
def load_keras_model(keras_file,custom_import=False,whetstone=False,larq=False):
    print('Loading Keras model...')

    if custom_import:
        from tensorflow.keras.utils import CustomObjectScope
        from helpers.custom_layers.clip_constraints import ClipPrecision, Clip
        if whetstone:
            from helpers.custom_layers.layers_bnn import Softmax_Decode,Spiking_BRelu
            customObjects = {'Spiking_BRelu':Spiking_BRelu,'ClipPrecision':ClipPrecision,'Clip':Clip,'Softmax_Decode':Softmax_Decode}
        elif larq:
            import larq
            customObjects = {'QuantConv2D':larq.layers.QuantConv2D,'QuantDense':larq.layers.QuantDense}
        else:
            customObjects = {'ClipPrecision':ClipPrecision}
        with CustomObjectScope(customObjects):
            model = load_model(keras_file, compile=False)
    else:
        model = load_model(keras_file, compile=False)

    return model