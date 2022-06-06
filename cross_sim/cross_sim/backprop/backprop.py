#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import time
import random
from math import sqrt
import numpy as np
import os
import pickle
from .activate import Activate
from .ncore import Ncore
from .ncore_new import NcoreXbar
from numbers import Number
from .backprop_util import apply_pool, flatten_layer, space_to_depth, apply_quantization, init_GPU_util, decode_from_key
from .activate import SIGMOID, SIGMOIDSLOPE, RECTLINEAR, SOFTMAX, SHIFTED_SIGMOID,WHETSTONE, QUANTIZED_RELU, NONE, STYLES
from ..xbar_simulator.parameters.parameter_defaults import WriteNoiseModelEnum, UpdateModelEnum, CrossbarTypeEnum

# precision used for all Numpy matrices, vectors
precision = np.float32

class Backprop:
    """
  This class creates a multi-layer neural network, reads an input data
  set, and uses the network to either perform classification with
  pre-trained weights, or to perform training and optimize the weights.
  """

    def __init__(self, layers, seed=None):
        """
        define a backprop network
        required args:
          layers defined for a convolutional network as a tuple of tuples  (or list of lists). each inner tuple represents: (x size, y size, channels)
                Thus we have:( (x_in,y_in,channel_in),(x_hidden1,y_hidden1,channel_hidden1), ...,(x_out,y_out,channel_out))
                fully connected layers should be specified as:
                    ( (1,1,n_in),(1,1,n_hidden1), ...,(1,1,n_out)) OR ( n_in,n_hidden1, ...,n_out)
        """

        # For old style fully connected networks append (1,1,number) to layer specification
        self.layers = []
        for layer in layers:
            if isinstance(layer, Number):
                self.layers.append((1,1,layer) )
            else:
                self.layers.append(layer)

        self.nlayer = len(layers) - 1
        self.ndata = 0
        self.indata = None
        self.answers = None
        self.alpha = 0.01  # learning rate for training

        # default activation function
        # can be redefined by activate()

        self.activate = Activate(style="SIGMOID")
        self.activate_output = Activate(style="SIGMOID")
        self.costFunction = None

        # Set a separate activation function for each layer
        # Final layer still uses activate_output
        self.separate_activations = True
        self.activations = [None for k in range(self.nlayer-1)]
        for k in range(self.nlayer-1):
            self.activations[k] = Activate(style="SIGMOID")
        self.layerTypes = None

        # Batch norm
        self.batch_norm = [False for layer in range(self.nlayer)]
        self.batch_norm_params = [None for layer in range(self.nlayer)]
        self.batch_norm_epsilons = [1e-3 for layer in range(self.nlayer)]

        # Parameters for non MVM active layers (pool, add, concat)
        self.auxLayerParams = self.nlayer * [None]

        # ID of the layer that sources each layer (necessary for ResNets, etc)
        self.sourceLayers = self.nlayer * [None]
        self.memory_window = 0

        # cores are defined by call to ncore()
        self.ncore_style = self.nlayer * [None]  # create list of length nlayer
        self.ncores = self.nlayer * [0]  # create list of length nlayer

        self.whetstone = False
        self.decoder_mat = None
        self.digital_bias = self.nlayer * [False]
        self.bias_values = self.nlayer * [None]
        self.scale_values = self.nlayer * [None]
        self.quantization_values = self.nlayer * [None]
        self.useGPU = False
        self.gpu_id = 0
        self.profile_DAC_inputs = False
        self.profile_ADC_inputs = False
        self.profile_ADC_biased = False

        self.lr_sched = False
        self.lr_sched_function = False

        # Set CPU as default for utility functions
        init_GPU_util(False)

        # initialize Numpy and Python RNGs
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed + 1)

    # -------------------------------------------------------
    # Define neural core, activation, noise, precision models
    # -------------------------------------------------------

    # create one or more neural cores
    # style = "conv", "new", or "new_bias"
    #   for ideal numeric, "old" or "old_bias" (not applicable to inference)
    # which = 0 to create all cores
    # which = 1 to N to create Nth core

    def ncore(self, style="new", **kwargs):
        if "which" in kwargs:
            which = kwargs["which"]
        else:
            which = 0
        self.ncore_style[which - 1] = style
        if which:
            self.ncores[which - 1] = self.create_onecore(style, which - 1, **kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i] = self.create_onecore(style, i, **kwargs)

    # create Ith core, I = 0 to nlayer-1
    # called by ncore(), not by user script

    def create_onecore(self, style, i, **kwargs):

        if style=="conv":
            # Create a convolutional layer
            # The row and column numbers here will be ignored: an appropriately sized matrix
            # will be initialized based on convolutional layer parameters (params) in NcoreXbar
            # A bias is handled internally inside the convolution object
            ncore = NcoreXbar(0, 0, precision, **kwargs)
        else:
            # If core is not a conv layer, need to check that specified inputs and outputs are indeed vectors
            if self.layers[i][0] != 1 and self.layers[i][1] != 1:
                raise ValueError("Invalid size of input to a fully connected layer")
            if self.layers[i+1][0] != 1 and self.layers[i+1][1] != 1:
                raise ValueError("Invalid size of output from a fully connected layer")

            if style == "old": # ideal numeric no bias
                ncore = Ncore(self.layers[i+1][2], self.layers[i][2], precision, **kwargs)
            elif style == "old_bias": # ideal numeric with bias
                ncore = Ncore(self.layers[i+1][2], self.layers[i][2]+1, precision, **kwargs)
            elif style == "new": # crossbar no bias
                ncore = NcoreXbar(self.layers[i+1][2], self.layers[i][2], precision, **kwargs)
            elif style == "new_bias": # crossbar with bias
                ncore = NcoreXbar(self.layers[i+1][2], self.layers[i][2]+1, precision, **kwargs)
            else:
                error("Unknown neural core style")

        return ncore

    def set_activate(self, **kwargs):
        # Set the same activation for all layers at once, except output
        # This function is mostly kept for backward compatibility
        self.activate = Activate(**kwargs,useGPU=self.useGPU)
        self.separate_activations = True
        for i in range(self.nlayer-1):  
            self.activations[i] = Activate(**kwargs,useGPU=self.useGPU)

    def set_activate_output(self, **kwargs):
        self.activate_output = Activate(**kwargs,useGPU=self.useGPU)
        # If cost function has not been set yet, set it to a default option
        if self.costFunction is None:
            if self.activate_output.style == SOFTMAX:
                self.costFunction = "cross_entropy"
            else:
                self.costFunction = "L2"

    def set_activations(self,layer,**kwargs):
        self.separate_activations = True
        self.activations[layer] = Activate(**kwargs,useGPU=self.useGPU)

    def set_learning_rate(self,alpha0,lr_sched,lr_sched_function):
        self.alpha = alpha0
        self.lr_sched = lr_sched
        self.lr_sched_function = lr_sched_function

    def readnoise(self, which=0, **kwargs):
        if which:
            self.ncores[iwhich - 1].set_readnoise(**kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i].set_readnoise(**kwargs)

    def writenoise(self, which=0, **kwargs):
        if which:
            self.ncores[iwhich - 1].set_writenoise(**kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i].set_writenoise(**kwargs)

    def nonlinear(self, which=0, **kwargs):
        if which:
            self.ncores[iwhich - 1].set_nonlinear(**kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i].set_nonlinear(**kwargs)

    def a2d(self, which=0, **kwargs):
        if which:
            self.ncores[iwhich - 1].set_a2d(**kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i].set_a2d(**kwargs)

    def d2a(self, which=0, **kwargs):
        if which:
            self.ncores[iwhich - 1].set_d2a(**kwargs)
        else:
            for i in range(self.nlayer):
                self.ncores[i].set_d2a(**kwargs)

    # -------------------------------------------------------
    # Matrix weights and input data
    # -------------------------------------------------------

    def read_weights(self, files, verbose=False):
        """
        populate Ncore weight matrices with pre-computed weights
        files = text files with weight values
        return list of Numpy matrices
        """

        if len(files) != self.nlayer: error("File count does not match layers")

        for ilayer in range(self.nlayer):
            lines = open(files[ilayer], 'r').readlines()
            firstline = lines.pop(0)
            two = firstline.split()
            #nrow, ncol = int(two[0]), int(two[1])
            nrow, ncol = int(two[2][:-1]), int(two[5][:-1])
            if self.ncore_style[ilayer] == "old_bias" or self.ncore_style[ilayer] == "new_bias":
                ncol += 1
            if ncol != self.ncores[ilayer].ncol:
                error("Weight file column count does not match defined layer size")
                if nrow != self.ncores[ilayer].nrow:
                    error("Weight file row count does not match defined layer size")

            # set_matrix() will convert matrix to precision
            matrix = np.zeros((nrow, ncol))
            for line in lines:
                values = line.split()
                i = int(values[0])
                j = int(values[1])
                matrix[i, j] = float(values[2])

            self.ncores[ilayer].set_matrix(matrix, applyErrors=True, verbose=verbose)

        matrices = []
        for ilayer in range(self.nlayer):
            matrices.append(self.ncores[ilayer].get_matrix())

        return matrices

    def read_weights_npz(self, dict_in, verbose=False):
        """
        populate Ncore weight matrices with pre-computed weights
        use with whetstone trained MLPs
        return list of Numpy matrices
        """

        for ilayer in range(self.nlayer):
            if ilayer == 0:
                mat = dict_in['layer0']
                nrow, ncol = int(mat.shape[1]), int(mat.shape[0])
            else:
                mat = dict_in['layer1']
                nrow, ncol = int(mat.shape[1]), int(mat.shape[0])
            if ncol != self.layers[ilayer][2]:
                error("Weight file column count does not match defined layer size")
                if nrow != self.layers[ilayer + 1]:
                    error("Weight file row count does not match defined layer size")
            matT = mat.T
            self.ncores[ilayer].set_matrix(matT, applyErrors=True, verbose=verbose)

        matrices = []
        for ilayer in range(self.nlayer):
            matrices.append(self.ncores[ilayer].get_matrix())
        return matrices


    def read_weights_crossSim(self, loadPath, verbose=False):
        """
        populate Ncore weight matrices with pre-trained weights in CrossSim written using write_weights_npz
        loadPath = path to npz file
        return list of Numpy matrices
        """

        matrices_list = np.load(loadPath,allow_pickle=True)['mats']
        for ilayer in range(self.nlayer):
            mat = matrices_list[ilayer]
            nrow, ncol = mat.shape
            if ncol != self.ncores[ilayer].ncol:
                raise ValueError("Weight file column count does not match defined layer size")
            if nrow != self.ncores[ilayer].nrow:
                raise ValueError("Weight file row count does not match defined layer size")
            self.ncores[ilayer].set_matrix(mat, applyErrors=True, verbose=verbose)
        matrices = []
        for ilayer in range(self.nlayer):
            matrices.append(self.ncores[ilayer].get_matrix())

        return matrices


    def read_weights_keras(self,weight_dict,layerParams,fold_batchnorm=False,verbose=False):
        """
        populate Ncore weight matrices from a Keras model file
        weight_dict: dictionary container of weights from Keras model indexed by layer name
        """

        # whetstone decoder mat
        if self.whetstone:
            last_key = list(weight_dict)[-1]
            self.decoder_mat = weight_dict[last_key][0]
            
        for k in range(len(layerParams)):
            if layerParams[k]['batch_norm'] is None:
                self.batch_norm[k] = False
            else:
                if self.layerTypes[k] in ("conv","dense") and fold_batchnorm:
                    self.batch_norm[k] = False
                else:
                    self.batch_norm[k] = True

        for ilayer in range(self.nlayer):

            # Extract batch norm if applicable
            if self.batch_norm[ilayer]:
                Wbn = weight_dict[layerParams[ilayer]['batch_norm']]
                if self.auxLayerParams[ilayer]['BN_scale'] and not self.auxLayerParams[ilayer]['BN_center']:
                    Wbn = [Wbn[0], np.zeros(Wbn[0].shape), Wbn[1], Wbn[2]]
                elif not self.auxLayerParams[ilayer]['BN_scale'] and self.auxLayerParams[ilayer]['BN_center']:
                    Wbn = [np.ones(Wbn[0].shape), Wbn[0], Wbn[1], Wbn[2]]
                elif not self.auxLayerParams[ilayer]['BN_scale'] and not self.auxLayerParams[ilayer]['BN_center']:
                    Wbn = [np.ones(Wbn[0].shape), np.zeros(Wbn[0].shape), Wbn[0], Wbn[1]]

                for j in range(len(Wbn)):
                    Wbn[j] = Wbn[j].astype(precision)
                    if self.layerTypes[ilayer] in ("conv","pool"):
                        Wbn[j] = Wbn[j].reshape((len(Wbn[j]),1,1))
                    if self.useGPU:
                        Wbn[j] = cp.array(Wbn[j])
                self.batch_norm_epsilons[ilayer] = self.auxLayerParams[ilayer]['epsilon']
                self.batch_norm_params[ilayer] = Wbn


            if self.layerTypes[ilayer] not in ("conv","dense"):
                continue

            # Extract raw tensors from Keras model
            Wi_0 = weight_dict[layerParams[ilayer]['name']]
            Wi = Wi_0[0].astype(precision)

            # Larq weight binarization
            if self.auxLayerParams[ilayer]['binarizeWeights']:
                Wi = np.sign(Wi)

            # Extract the bias weights
            if (self.ncore_style[ilayer] == "conv" and self.ncores[ilayer].params.convolution_parameters.bias) or \
                (self.ncore_style[ilayer] == "old_bias" or self.ncore_style[ilayer] == "new_bias"):
                useBias = True
                Wbias = Wi_0[1].astype(precision)
            else:
                useBias = False

            matrix = np.zeros((self.ncores[ilayer].nrow, self.ncores[ilayer].ncol),dtype=Wi.dtype)

            if self.ncore_style[ilayer] == "conv":
                Kx,Ky,Nic,Noc = Wi.shape

                if self.ncores[ilayer].params.convolution_parameters.depthwise:
                    Noc = Nic

                # Check dimensions
                if Noc != matrix.shape[0]:
                    raise ValueError("Mismatch in conv layer along output dimension")
                if Kx != self.ncores[ilayer].params.convolution_parameters.Kx:
                    raise ValueError("Mismatch in kernel size Kx")
                if Ky != self.ncores[ilayer].params.convolution_parameters.Ky:
                    raise ValueError("Mismatch in kernel size Kx")
                if Nic != self.ncores[ilayer].params.convolution_parameters.Nic:
                    raise ValueError("Mismatch in input channel size: "+str(Nic)+" vs "+str(self.ncores[ilayer].params.convolution_parameters.Nic))
                if (useBias and int(Kx*Ky*Nic+1) != matrix.shape[1]) or (not useBias and int(Kx*Ky*Nic) != matrix.shape[1]):
                    raise ValueError("Mismatch in conv layer along input dimension")

                if not self.ncores[ilayer].params.convolution_parameters.depthwise:
                    for i in range(Noc):
                        submat = Wi[:,:,:,i]
                        submat_flat = np.array([submat[:,:,k].flatten() for k in range(Nic)])
                        submat_flat = submat_flat.flatten()
                        if useBias:
                            matrix[i,:-1] = submat_flat
                        else:
                            matrix[i,:] = submat_flat
                    if useBias:
                        matrix[:,-1] = Wbias
                
                else:
                    for i in range(Noc):
                        submat = Wi[:,:,i,0]
                        submat_flat = submat.flatten()
                        matrix[i,(i*Kx*Ky):((i+1)*Kx*Ky)] = submat_flat
                    if useBias:
                        matrix[:,-1] = Wbias
            else:
                Wi = Wi.transpose()
                if not useBias:
                    if matrix.shape != Wi.shape:
                        raise ValueError("Mismatch in FC layer dimensions (no bias)")
                    matrix = Wi
                elif useBias:
                    if (matrix.shape[0] != Wi.shape[0]) or (matrix.shape[1] != (Wi.shape[1]+1)):
                        raise ValueError("Mismatch in FC layer dimensions (with bias)")
                    matrix[:,:-1] = Wi
                    matrix[:,-1] = Wbias

            self.ncores[ilayer].set_matrix(matrix, applyErrors=True, verbose=verbose)


    def write_weights(self, files):
        """
        write weight matrices to text files
        """
        if len(files) != self.nlayer: error("File count does not match layers")

        for ilayer in range(self.nlayer):

            matrix = self.ncores[ilayer].get_matrix()

            if self.ncore_style[ilayer] == "new" or self.ncore_style[ilayer] == "new_bias":
                if self.useGPU:
                    import cupy as cp
                    cp.cuda.Device(self.gpu_id).use()
                    matrix = cp.asnumpy(matrix)

            with open(files[ilayer], 'w') as lines:
                lines.write('{0} {1}\n'. \
                            format(self.layers[ilayer + 1], self.layers[ilayer]))
                it = np.nditer(matrix, flags=['multi_index'])
                while not it.finished:
                    lines.write('{0} {1} {2}\n'.format(it.multi_index[0],
                                                       it.multi_index[1], it[0]))
                    it.iternext()


    def write_weights_npz(self, savePath, verbose=False):
        """
        write weights to an npz archive
        """
        matrix_list = [None for i in range(self.nlayer)]

        for ilayer in range(self.nlayer):
            matrix = self.ncores[ilayer].get_matrix()
            if self.ncore_style[ilayer] == "new" or self.ncore_style[ilayer] == "new_bias":
                if self.useGPU:
                    import cupy as cp
                    cp.cuda.Device(self.gpu_id).use()
                    matrix = cp.asnumpy(matrix)
            matrix_list[ilayer] = matrix.copy()

        # Save matrices along with some metadata (learning rate and activations)
        #np.savez(savePath,mats=matrix_list,alpha=self.alpha,activate=self.activate,activate_output=self.activate_output)
        #saving the additional metadata seems to be breaking some things for the simple weight export, we will need to fix this later
        np.savez(savePath, mats=matrix_list)


    def random_weights(self, scale="none", spread=1.0, verbose=False,pos_init = False):
        """
        populate Ncore weight matrices with random weights from -spread to spread
        scale options = "none" or "bengio"
        return list of Numpy matrices
        """
        # matrix created at 64-bit precision
        # set_matrix() will convert to precision
        for ilayer in range(self.nlayer):
            size = (self.ncores[ilayer].nrow, self.ncores[ilayer].ncol)
            if scale == "bengio":
                spread = 4.0 * sqrt(6.0/(size[0]+size[1]))
                # spread = 1 * sqrt(6.0 / (size[0] + size[1]))
                # matrix = np.random.uniform(0, spread, size)
            if pos_init == True:
                matrix = np.random.uniform(0,1,size)
            else:
                matrix = np.random.uniform(-spread,spread,size)
            if self.ncore_style[ilayer] == "old" or self.ncore_style[ilayer] == "old_bias":
                self.ncores[ilayer].set_matrix(matrix,  verbose=verbose)
            else:
                self.ncores[ilayer].set_matrix(matrix, applyErrors=False, verbose=verbose)

        matrices = []
        for ilayer in range(self.nlayer):
            matrices.append(self.ncores[ilayer].get_matrix())
        return matrices


    def read_inputs(self, file, scale="max", scaleparam=None,
                    scalereturn=0, shuffle=0, verbose=True, data=None):
        """
        read inputs and answers from file
        store at precision
        scale options:
          applied to "matrix" of datums x features
          max = divide entire matrix by max feature over entire matrix
          colmax = find max in each column
            divide all elements in each column by that column's value
          colgauss = do following for each column:
            find mean, shift so mean = 0.0
            scale around new mean so 0.5 sigma (use sigma=0.5 so that range of -1 to 1 captures most data)
          gauss = do following for the entire matrix (not by column):
            find mean, shift so mean = 0.0
            scale around new mean so 0.5 sigma

          none = perform no scaling
        scale input matrices by max value so become 0.0 to 1.0
        return # of datums, array of inputs, vector of answers
        if verbose is true, print the scaling
        """
        if data is None:
            data = np.loadtxt(file, delimiter=',', dtype=precision)

        inputLayerSize = self.layers[0][0]*self.layers[0][1]*self.layers[0][2]
        if data.shape[1] != inputLayerSize + 1:
            error("Size of input datums not compatible with defined input size")

        if shuffle: np.random.shuffle(data)

        self.ndata = data.shape[0]
        self.indata = data[:, :-1]
        self.answers = data[:, -1]

        if scale == "max":
            if scaleparam:
                factor = scaleparam
            else:
                factor = np.amax(np.fabs(self.indata))
            self.indata /= factor
            scaleinfo = factor
            if verbose:
                print("Max scaling: shrink by", factor)
        elif scale == "colmax":
            if scaleparam:
                colfactor = scaleparam
            else:
                colfactor = np.amax(np.fabs(self.indata), axis=0)
            self.indata /= colfactor
            scaleinfo = colfactor
            if verbose:
                print("Colmax scaling: max shrink of any column by", np.amax(factor))
        elif scale == "colgauss":
            if scaleparam:
                mean, std = scaleparam
                self.indata -= mean
                self.indata /= std
            else:
                mean = np.mean(self.indata,
                               axis=0)  # TODO:   *************** remove axis =0?? (takes mean per col vs per dataset)
                self.indata -= mean
                std = np.std(self.indata, axis=0) * 2  # factor of 2 results in scaling to sigma = 0.5
                self.indata /= std
            scaleinfo = (mean, std)
            if verbose:
                print("Colgauss scaling: max column mean, std =",
                      np.amax(np.fabs(mean)), np.amax(std))
        elif scale == "gauss":
            if scaleparam:
                mean, std = scaleparam
                self.indata -= mean
                self.indata /= std
            else:
                mean = np.mean(self.indata)
                self.indata -= mean
                std = np.std(self.indata) * 2
                self.indata /= std
            scaleinfo = (mean, std)
            if verbose:
                print("Gauss scaling: max column mean, std =",
                      mean, std)

        elif scale == "none":
            if verbose: print("No scaling: inputs unchanged")
        else:
            error("Invalid scale parameter")

        if scalereturn: return self.ndata, self.indata, self.answers, scaleinfo
        return self.ndata, self.indata, self.answers


    def random_inputs(self, n):
        """
        generate N random inputs and answers
        store at precision
        each input is random values from 0 to 1
        each answer is random integer from 0 to self.noutput-1 inclusive
        return # of datums, array of inputs, vector of answers
        """

        self.ndata = n

        # NOET: could do this via Numpy rand() function

        indata = []
        answers = []
        inputLayerSize = self.layers[0][0]*self.layers[0][1]*self.layers[0][2]
        for i in range(n):
            vec = []
            for j in range(inputLayerSize):
                vec.append(random.random())
            indata.append(vec)
            answers.append(random.randrange(0, self.noutput))

        self.indata = np.array(indata, precision)
        self.answers = np.array(answers, precision)
        return self.ndata, self.indata, self.answers


    def init_GPU(self,useGPU,gpu_id):
        global ncp
        self.useGPU = useGPU
        if useGPU:
            global cp
            import cupy as cp
            self.gpu_id = gpu_id
            cp.cuda.Device(gpu_id).use()
            ncp = cp
        else:
            ncp = np

        init_GPU_util(useGPU)


    def expand_cores(self):
        """
        Duplicate the arrays inside the neural cores in order to allow parallel sliding window computation of convolutions
        Duplication factor is set by x_par and y_par
        """
        if self.ncore_style[0] != "old" and self.ncore_style[0] != "old_bias":
            for ilayer in range(self.nlayer):
                if self.layerTypes[ilayer] == "conv":
                    Ncopy = self.ncores[ilayer].params.numeric_params.x_par * self.ncores[ilayer].params.numeric_params.y_par
                    if self.ncores[ilayer].neural_core.Ncores == 1:
                        # Let each core know its layer number
                        self.ncores[ilayer].neural_core.core.layer_number = ilayer
                        self.ncores[ilayer].neural_core.core.expand_matrix(Ncopy,mode=0)
                    else:
                        for k in range(self.ncores[ilayer].neural_core.Ncores):
                            self.ncores[ilayer].neural_core.cores[k].layer_number = ilayer
                            self.ncores[ilayer].neural_core.cores[k].expand_matrix(Ncopy,mode=0)


    def unexpand_cores(self):
        """
        Undo expand_cores() to free up memory
        """
        if self.ncore_style[0] != "old" and self.ncore_style[0] != "old_bias":
            for ilayer in range(self.nlayer):
                if self.layerTypes[ilayer] == "conv" and \
                (self.ncores[ilayer].params.numeric_params.x_par > 1 or self.ncores[ilayer].params.numeric_params.y_par > 1):
                    if self.ncores[ilayer].neural_core.Ncores == 1:
                        self.ncores[ilayer].neural_core.core.unexpand_matrix()
                    else:
                        for k in range(self.ncores[ilayer].neural_core.Ncores):
                            self.ncores[ilayer].neural_core.cores[k].unexpand_matrix()


    # -------------------------------------------------------
    # Classification
    # -------------------------------------------------------

    def classify(self, n=0, debug=0,count_interval=0,randomSampling=True,topk=1,time_interval=False,profiling_folder=None,profiling_settings=[False,False,False]):
        """
        perform a forward calculation on set of inputs
        use stored weight matrices
        n = # of inputs to process, 0 = default = all
        compare each result to true answer
        return count & fraction of results that are correct
    
        """

        if not self.ncore_style: raise ValueError("Neural core is not initialized")
        if not n: n = self.ndata
        if n > self.ndata: 
            warn("N too large for classification")
            n = self.ndata

        # DAC inputs are always profiled inside backprop in algorithmic units
        # ADC inputs are profiled inside backprop if input bit slicing and weight bit slicing are disabled, in algorithmic units
        # Otherwise, it is profiled inside the WrapperCore object (e.g. balanced_core, offset_core, bitsliced_core), in xbar units
        profile_in_backprop = False
        self.profile_DAC_inputs, self.profile_ADC_inputs, self.profile_ADC_biased = profiling_settings
        if self.profile_DAC_inputs:
            profile_in_backprop = True
        elif self.profile_ADC_inputs:
            if not self.ncores[0].params.xbar_params.input_bitslicing and self.ncores[0].params.xbar_params.Nslices == 1:
                profile_in_backprop = True
                
        if profile_in_backprop or self.profile_ADC_inputs:
            if profiling_folder is None:
                raise ValueError("Please provide a valid folder to save profiling results to")
            self.profiled_values = [ncp.zeros(0,dtype=ncp.float32) for i in range(self.nlayer)]

        # Depending on the datatype for topk, output is either an integer or an array of accuracies
        if type(topk) is int:
            count = 0
        else:
            count = np.zeros(len(topk))

        # Generate a list of random example numbers
        if n == self.ndata: randomSampling = False
        if randomSampling:
            print("Randomized data set")
            inds_rand = np.array(random.sample(range(0,self.ndata),n))

        # Check if network branches, e.g. ResNet
        # This is true if sourceLayers has been set
        branch = (self.sourceLayers != self.nlayer * [None])

        if time_interval:
            t_avg, one_T = 0, 0
            T1 = time.time()

        # Classify the images
        for one in range(n):
            # Display the cumulative accuracy and compute time to the user
            if one > 0 and (count_interval > 0) and one % count_interval == 0:
                time_msg, accs = "", ""
                if time_interval:
                    t_avg = t_avg*(one_T/(one_T+1)) + (time.time()-T1)/(one_T+1)
                    time_msg = ', time = {:.4f}'.format(t_avg)+'s'
                    T1, one_T = time.time(), one_T + 1
                if type(topk) is int:
                    accs = "{:.2f}".format(100*float(count)/one) + "%"
                else:
                    for j in range(len(topk)):
                        accs += "{:.2f}".format(100*float(count[j])/one) + "% (top-" + str(topk[j]) + ")"
                        if j < (len(topk)-1): accs += ", "
                print("Example " + str(one)+"/"+str(n) + ", accuracy so far = " + accs + time_msg,end='\r')

            ex = (one if not randomSampling else inds_rand[one])
            result = self.classify_one(ex, branch, debug=debug,topk=topk,profile_in_backprop=profile_in_backprop)

            if type(topk) is int:
                count += result
            else:
                for j in range(len(topk)):
                    count[j] += result[j]

        if count_interval > 0:
            print('\n')

        # Number of cores for each layer
        Ncores = np.zeros(self.nlayer,dtype=int)
        if self.ncore_style[0] != "old" and self.ncore_style[0] != "old_bias":
            for ilayer in range(self.nlayer):
                if self.layerTypes[ilayer] == "conv":
                    Ncores[ilayer] = self.ncores[ilayer].neural_core.Ncores
                elif self.layerTypes[ilayer] == "dense":
                    Ncores[ilayer] = self.ncores[ilayer].Ncores

        # Save the profiled activations to the destination folder
        for ilayer in range(self.nlayer):

            if self.layerTypes[ilayer] not in ("conv","dense") or self.ncore_style[ilayer] == "old" or self.ncore_style[ilayer] == "old_bias":
                continue

            # Save profiled DAC/ADC inputs that are stored in backprop object
            if profile_in_backprop:
                if ilayer == 0:
                    print("Saving DAC or ADC input data")
                if self.profile_ADC_inputs:
                    np.save(profiling_folder+'adc_inputs_layer'+str(ilayer)+'.npy',self.profiled_values[ilayer])
                elif self.profile_DAC_inputs:
                    np.save(profiling_folder+'dac_inputs_layer'+str(ilayer)+'.npy',self.profiled_values[ilayer])

            # Save profiled ADC input values stored inside cores, rather than in backprop
            # These structures are resolved along several dimensions:
            #   Cores: concatenated in same file
            #   Input bits:
            #       If not weight bit slicing, save in separate files
            #       If weight bit slicing, input bits are not resolved inside core, so concatenate in same file
            #   Weight slices: saved in separate files 
            elif self.ncores[ilayer].params.xbar_params.profile_ADC_inputs:
                if ilayer == 0:
                    print("Saving ADC input data")

                # Get handle to neural core or cores list
                if Ncores[ilayer] == 1:
                    if self.layerTypes[ilayer] == "conv":
                        core = self.ncores[ilayer].neural_core.core
                    elif self.layerTypes[ilayer] == "dense":
                        core = self.ncores[ilayer].neural_core
                else:
                    if self.layerTypes[ilayer] == "conv":
                        cores = self.ncores[ilayer].neural_core.cores
                    elif self.layerTypes[ilayer] == "dense":
                        cores = self.ncores[ilayer].neural_cores

                # Profiling ADC inputs with input bit slicing but not weight bit slicing
                if self.ncores[ilayer].params.xbar_params.input_bitslicing and self.ncores[ilayer].params.xbar_params.Nslices == 1:
                    Nbits_in = core.array_outputs.shape[1]
                    if Ncores[ilayer] == 1:
                        for k in range(Nbits_in):
                            np.save(profiling_folder+'adc_inputs_layer'+str(ilayer)+'_ibit'+str(k)+'.npy',core.array_outputs[:,k,:].flatten())
                    else:
                        for k in range(Nbits_in):
                            all_core_outputs = ncp.zeros(0)
                            for icore in range(Ncores[ilayer]):
                                all_core_outputs = ncp.concatenate((all_core_outputs,cores[icore].array_outputs[:,k,:].flatten()))
                            np.save(profiling_folder+'adc_inputs_layer'+str(ilayer)+'_ibit'+str(k)+'.npy',all_core_outputs)

                # For weight bit slicing, ADC inputs are resolved by weight bit slice but not input bit
                elif self.ncores[ilayer].params.xbar_params.Nslices > 1:
                    if Ncores[ilayer] == 1:
                        for k in range(self.ncores[ilayer].params.xbar_params.Nslices):
                            np.save(profiling_folder+'adc_inputs_layer'+str(ilayer)+'_slice'+str(k)+'.npy',core.bitslice_outputs[:,k,:].flatten())
                    else:
                        for k in range(self.ncores[ilayer].params.xbar_params.Nslices):
                            bitslice_outputs_k = ncp.zeros(0)
                            for icore in range(Ncores[ilayer]):
                                bitslice_outputs_k = ncp.concatenate((bitslice_outputs_k,cores[icore].bitslice_outputs[:,k,:].flatten()))
                            np.save(profiling_folder+'adc_inputs_layer'+str(ilayer)+'_slice'+str(k)+'.npy',bitslice_outputs_k)
        
        return count, count / n


    def classify_one(self, one, branch, debug=0,profile_in_backprop=False,topk=1,debug_graph=False):
        """
        Perform a forward calculation on a single input
        """

        if one < 0 or one >= self.ndata:
            error("Input index vector %d is out-of-bounds" % one)

        indata = self.indata
        answers = self.answers

        nlayer = self.nlayer
        ncores = self.ncores
        activate = self.activate
        layers = self.layers
        m_par = 0

        if branch:
            sourceLayers = self.sourceLayers
            output_vecs = self.nlayer * [None]
            output_vecs_add = self.nlayer * [None]
            memory_window = self.memory_window # number of layers to hold in memory before erasure

        splitMVMprofiling = [False]*nlayer
        if profile_in_backprop and self.profile_ADC_inputs:
            for m in range(nlayer):
                if self.layerTypes[m] in ("conv","dense"):
                    splitMVMprofiling[m] = (ncores[m].Ncores > 1)
        
        for m in range(nlayer):

            if branch and debug_graph:
                print('Layer '+str(m)+': '+self.layerTypes[m])
                if m > 0:
                    print('   Source 1: layer '+str(sourceLayers[m][0]))
                    if len(sourceLayers[m])>1:
                        print('   Source 2: layer '+str(sourceLayers[m][1]))
                print('   Batchnorm: '+str(self.batch_norm[m]))
                if m != nlayer-1:
                    print('   Activation: '+str(STYLES[self.activations[m].style]))
                else:
                    print('   Activation: '+str(STYLES[self.activate_output.style]))

            if m == 0:
                ivec = indata[one].copy()
                flattened = (ivec.shape[0] == 1 and ivec.shape[1] == 1) or self.layerTypes[0] == "dense"
            elif not branch:
                ivec = output_vec.copy()

            if self.layerTypes[m] in ("conv","dense"):
                
                if branch and m != 0:
                    m_src = sourceLayers[m][0]
                    try:
                        if self.layerTypes[m_src] == "add" and self.auxLayerParams[m]['splitBeforeBN']:
                            ivec = output_vecs_add[m_src].copy()
                        else:
                            ivec = output_vecs[m_src].copy()
                    except AttributeError:
                        raise ValueError("Insufficient memory window for the neural network")

                # Shape input to be compatible with a conv layer, if possible
                if self.layerTypes[m] == "conv":
                    mvec, mvec_all = ncores[m].apply_convolution(ivec,profiling=splitMVMprofiling[m])
                elif self.layerTypes[m] == "dense":
                    if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                        if self.useGPU:
                            ivec0 = cp.ones(ivec.size+1,dtype=ivec.dtype)
                            ivec0[:-1] = ivec.flatten()
                            ivec = ivec0
                        else:
                            ivec = np.append(ivec, 1)
                    if not splitMVMprofiling[m]:
                        mvec = ncores[m].matvec(ivec)
                        mvec_all = None
                    else:
                        mvec, mvec_all = ncores[m].matvec(ivec,profiling=True)

                if profile_in_backprop:
                    self.update_neuron_statistics(m,ivec,mvec,mvec_all)

                # Apply digital bias
                if self.digital_bias[m]:
                    mvec += self.bias_values[m]

                # Apply batch normalization
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        mvec = gamma*(mvec-mu)/ncp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        mvec = gamma*(mvec-mu)/(var + epsilon) + beta

                if m!=(nlayer-1):
                    output_vec = self.activations[m].apply(mvec)
                else:
                    output_vec = self.activate_output.apply(mvec)
                
                if branch:
                    output_vecs[m] = output_vec

            elif self.layerTypes[m] == "pool":
                MPx = self.auxLayerParams[m]['MPx']
                MPy = self.auxLayerParams[m]['MPy']
                px_L = self.auxLayerParams[m]['px_L']
                px_R = self.auxLayerParams[m]['px_R']
                py_L = self.auxLayerParams[m]['py_L']
                py_R = self.auxLayerParams[m]['py_R']
                stride_MP = self.auxLayerParams[m]['stride_MP']
                poolType = self.auxLayerParams[m]['poolType']
                avgPool_round = self.auxLayerParams[m]['round']
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    pvec = apply_pool(ivec,MPx,MPy,stride_MP,poolType,px_L,px_R,py_L,py_R,avgPool_round,self.useGPU)
                else:
                    pvec = apply_pool(ivec,MPx,MPy,stride_MP,poolType,px_L,px_R,py_L,py_R,avgPool_round,self.useGPU)

                # Apply batch normalization (rarely used)
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        pvec = gamma*(pvec-mu)/ncp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        pvec = gamma*(pvec-mu)/(var + epsilon) + beta

                # Apply activation to output (rarely used)
                if m!=(nlayer-1):
                    output_vec = self.activations[m].apply(pvec)
                else:
                    output_vec = self.activate_output.apply(pvec)

                if branch:
                    output_vecs[m] = output_vec

            # This layer type is not currently being supported in keras_parser()
            elif self.layerTypes[m] == "space2depth":
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    output_vecs[m] = space_to_depth(ivec,2,self.useGPU)
                else:
                    output_vec = space_to_depth(ivec,2,self.useGPU)

            elif self.layerTypes[m] == "add":

                Nsources = len(sourceLayers[m])
                if Nsources == 2:
                    m_src0 = sourceLayers[m][0]
                    m_src1 = sourceLayers[m][1]
                    if self.layerTypes[m_src0] == "add" and self.auxLayerParams[m]['splitBeforeBN']:
                        ivec0 = output_vecs_add[m_src0].copy()
                    else:
                        ivec0 = output_vecs[m_src0].copy()
                    if self.layerTypes[m_src1] == "add" and self.auxLayerParams[m]['splitBeforeBN']:
                        ivec1 = output_vecs_add[m_src1].copy()
                    else:
                        ivec1 = output_vecs[m_src1].copy()
                    mvec = ivec0 + ivec1
                else:
                    m_src0 = sourceLayers[m][0]
                    ivec0 = output_vecs[m_src0]
                    mvec = ivec0.copy()
                    for q in range(1,Nsources):
                        m_src = sourceLayers[m][q]
                        mvec += output_vecs[m_src]

                output_vecs_add[m] = mvec.copy()

                # Apply batch normalization
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        mvec = gamma*(mvec-mu)/ncp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        mvec = gamma*(mvec-mu)/(var + epsilon) + beta

                # Apply activation
                if m!=(nlayer-1):
                    output_vecs[m] = self.activations[m].apply(mvec)
                else:
                    output_vecs[m] = self.activate_output.apply(mvec)

            elif self.layerTypes[m] == "concat":

                Nsources = len(sourceLayers[m])
                m_src0 = sourceLayers[m][0]
                ivec0 = output_vecs[m_src0].copy()
                ovec = ivec0.copy()
                for q in range(1,Nsources):
                    m_src = sourceLayers[m][q]
                    ivec_q = output_vecs[m_src].copy()
                    # Check shapes
                    if ivec0.shape[1] != ivec_q.shape[1] or ivec0.shape[2] != ivec_q.shape[2]:
                        raise ValueError("Concat shapes incompatible")
                    ovec = ncp.concatenate((ovec,ivec_q),axis=0)

                output_vecs[m] = ovec

            elif self.layerTypes[m] == "quantize":
                shift_bits = self.auxLayerParams[m]['shift_bits']
                output_bits = self.auxLayerParams[m]['output_bits']
                signed = self.auxLayerParams[m]['signed']
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    output_vecs[m] = apply_quantization(ivec,self.quantization_values[m],shift_bits,output_bits,signed,self.useGPU)
                else:
                    output_vec = apply_quantization(ivec,self.quantization_values[m],shift_bits,output_bits,signed,self.useGPU)

            elif self.layerTypes[m] == "scale":
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()

                if m != (nlayer-1):
                    output_vec = ivec / self.scale_values[m]
                else:
                    output_vec = self.activate_output.apply(ivec / self.scale_values[m])

                if branch:
                    output_vecs[m] = output_vec

            # Assume flatten is applied just before the first dense layer
            if self.layerTypes[m] != "dense" and (m != nlayer-1) and self.layerTypes[m+1] == "dense" and not flattened:
                if branch:
                    output_vecs[m] = flatten_layer(output_vecs[m],self.useGPU)
                else:
                    output_vec = flatten_layer(output_vec,self.useGPU)
                flattened = True

            # Clear unneeded activations from memory
            if branch and memory_window > 0 and m >= memory_window:
                output_vecs[m-memory_window] = None
                output_vecs_add[m-memory_window] = None

        # Network output
        if branch:
            network_output = output_vecs[-1]
        else:
            network_output = output_vec

        if not flattened:
            network_output = flatten_layer(network_output,self.useGPU)
        if self.whetstone:
            if self.useGPU:
                network_output = cp.asnumpy(network_output)
            network_output = np.asarray(decode_from_key(self.decoder_mat, network_output))

        # Answer
        actual = int(answers[one])

        # Top-k accuracy:
        # If topk is an integer, returns a single value for top-k accuracy
        # If topk is a list, tuple, or array, returns a list of accuracies for different k
        if len(network_output) > 1:
            if type(topk) is int:
                if topk == 1:
                    index = network_output.argmax()
                    result = (1 if index == actual else 0)
                else:
                    indices = np.argpartition(network_output,-topk)[-topk:]
                    result = (1 if actual in indices else 0)
            else:
                result = np.zeros(len(topk))
                for j in range(len(topk)):
                    indices = np.argpartition(network_output,-topk[j])[-topk[j]:]
                    result[j] = (1 if actual in indices else 0)
        else:
            # Single output neuron case (for binary classification tasks)
            output = (0 if network_output < 0.5 else 1)
            result = (1 if output == actual else 0)

        return result


    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    def train(self, n=0, niter=1, out=0, minmax=0, i_epoch=0, debug=0, updateHistory=False):
        """
        perform backprop training on inputs
        stored matrices are adjusted as training proceeds
        n = # of inputs to train on, 0 = default = all
        niter = # of training iterations to perform
        out = 1 for timing output
        minmax = 1 for output of matrix bounds
        updateHistory : set to True to save the history of weights vs epoch
        """

        if not self.ncore_style: error("Neural core is not initialized")
        if not n: n = self.ndata
        if n > self.ndata: error("N too large for training")
        if n != self.ndata:
            n_vec = np.random.uniform(0,self.ndata-1,n)
            n_vec = np.round(n_vec).astype(int)

        indata = self.indata
        answers = self.answers

        nlayer = self.nlayer
        ncores = self.ncores
        activate = self.activate
        activate_output = self.activate_output

        alpha = self.alpha
        layers = self.layers

        if out: start = time.clock()

        ## Check whether to run the batch SIMD version of training
        # Nex_par is the number of examples to process in parallel (must be the same in all layers)
        if self.ncore_style[0] !="old" and self.ncore_style[0] !="old_bias":
            Nex_par = self.ncores[0].params.numeric_params.Nex_par
        else:
            Nex_par = 1

        if Nex_par > 1:
            trainSIMD = True
            for ilayer in range(self.nlayer):
                if self.ncores[ilayer].params.numeric_params.Nex_par != Nex_par:
                    trainSIMD = False
            if trainSIMD:
                return self.train_simd(n=n,niter=niter,out=out,minmax=minmax,batchsize=Nex_par,debug=debug)

        # forward steps
        if nlayer > 1:
            core_style = self.ncore_style[1]

        if updateHistory:
            # First core only
            self.weightHistory = np.zeros((n,np.size(self.ncores[0].get_matrix())))

        for i_iter in range(niter):
            aveerror = maxerror = 0.0
            if self.lr_sched:
                alpha = self.lr_sched_function(alpha,i_epoch)

            for one in range(n):
                
                input_vecs = nlayer * [0]  # list of input vectors at each level
                matvec_results = nlayer * [0]  # list of matvec results at each level
                output_vecs = nlayer * [0]  # list of output vectors at each level
                error_vecs = nlayer * [0]  # list of error vectors at each level

                # forward steps
                # save vectors before they get changed by A/D or D/A

                for m in range(nlayer):
                    if m == 0:
                        if n == self.ndata:
                            input_vec = indata[one].copy()
                        else:
                            ind = n_vec[one]
                            input_vec = indata[ind].copy()
                        #input_vec = ex.copy()
                    else:
                        input_vec = output_vecs[m - 1].copy()
                    if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                        if self.useGPU:
                            ivec0 = cp.ones(input_vec.size+1,dtype=input_vec.dtype)
                            ivec0[:-1] = input_vec.flatten()
                            input_vec = ivec0
                        else:
                            input_vec = ncp.append(input_vec, 1)
                    matvec_result = ncores[m].matvec(input_vec, debug=debug)

                    if m != (nlayer - 1):
                        output_vec = activate.apply(matvec_result)
                    else:
                        output_vec = activate_output.apply(matvec_result)

                    input_vecs[m] = input_vec
                    matvec_results[m] = matvec_result
                    output_vecs[m] = output_vec

                # Network output
                z = output_vecs[-1].copy()

                # Calculate loss and derivative of loss

                if layers[-1][2] > 1:
                    # Answer
                    y = ncp.zeros(len(z))
                    if n == self.ndata:
                        y[int(answers[one])]=1
                    else:
                        y[int(answers[ind])]=1

                elif layers[-1][2] == 1:
                    # Special case of one output neuron
                    if n == self.ndata:
                        y = int(answers[one])
                    else:
                        y = int(answers[ind])

                L1_err = y - z
                if self.costFunction == "L2":
                    error = 0.5*(L1_err)**2
                    error_vecs[nlayer-1] = L1_err
                elif self.costFunction == "cross_entropy":
                    epsilon = 1e-12 # prevent zero argument in logarithm or division
                    error = -(y*ncp.log(z+epsilon) + (1-y)*ncp.log(1-z+epsilon))
                    error_vecs[nlayer-1] = L1_err/(z*(1-z)+epsilon)

                # backward steps
                for m in range(nlayer - 1, -1, -1):
                    ### Do update, update = p*input_vec*alpha
                        # p = next layer backpropagated error (err) * derivative of activation fxn (dvec)
                        # input_vec = output of previous layer
                        # alpha = learning rate

                    # Final layer delta calculation is simplified in one of two cases:
                    # L2 loss and softmax (id = 3): delta is the L1 error
                    # Cross entropy loss and sigmoid (id = 0): delta is the L1 error

                    if m == (nlayer-1) and self.costFunction == "cross_entropy" and \
                    (self.activate_output.style == SIGMOID or self.activate_output.style == SOFTMAX):
                        p = L1_err
                    else:
                        dvec = activate.derivative(matvec_results[m])
                        p = error_vecs[m] * dvec

                    if m:
                        input_vec = output_vecs[m - 1].copy()
                        if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                            if self.useGPU:
                                ivec0 = cp.ones(input_vec.size+1,dtype=input_vec.dtype)
                                ivec0[:-1] = input_vec.flatten()
                                input_vec = ivec0
                            else:
                                input_vec = ncp.append(input_vec, 1)
                    else:
                        input_vec = input_vecs[0].copy()

                    # For PCM, use update burst in order to accommodate the need for occasional refresh
                    if (self.ncore_style[m] == "new_bias" or self.ncore_style[m] == "new") and\
                     ncores[m].params.numeric_params.update_model == UpdateModelEnum.PCM:
                        dW = ncp.outer(p,input_vec)
                        ncores[m].update_burst(dW, alpha, debug=debug)
                    else:
                        ncores[m].update(p, input_vec, alpha, debug=debug)

                    ## calculate error to send backwards
                    if m:
                        # self.plist.append(p)
                        error_vecs[m - 1] = ncores[m].vecmat(p, debug=debug)
                        if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                            error_vecs[m - 1] = error_vecs[m - 1][:-1]

                # Save weights after every update
                if updateHistory:
                    if not self.useGPU:
                        self.weightHistory[one,:] = ncores[m].get_matrix()[:]
                    else:
                        self.weightHistory[one,:] = ncores[m].get_matrix().asnumpy()[:]

                # update statistics on error
                error = np.mean(error)
                aveerror += error
                maxerror = max(maxerror, error)


            aveerror /= n
            if out: print(i_iter, aveerror, maxerror)

        if out:
            stop = time.clock()
            print("time =", stop - start, "secs");

        if minmax:
            for m in range(nlayer):
                wtmax = np.amax(ncores[m].get_matrix)
                wtmin = np.amin(ncores[m].get_matrix)
                print("Matrix %d: max1 %g min1 %g max2 %g min2 %g" % \
                      (wtmax, wtmin, m + 1))

        # Update learning rate
        self.alpha = alpha

        return aveerror, maxerror

    def train_batch_digital(self, n=0, niter=1, out=0, minmax=0, batchsize=10, i_epoch=0, debug=0):
        """
        perform minibatch training with a specified batch size. Weight updates are accumulated digitally
        and applied to the matrix in a single serial programming step

        stored matrices are adjusted as training proceeds
        n = # of inputs to train on, 0 = default = all
        niter = # of training iterations to perform
        out = 1 for timing output
        minmax = 1 for output of matrix bounds
        """

        if not self.ncore_style: error("Neural core is not initialized")
        if not n: n = self.ndata
        if n > self.ndata: error("N too large for training")
        #print(n)
        if n != self.ndata:
            n_vec = np.random.uniform(0,self.ndata-1,n)
            n_vec = np.around(n_vec).astype(int)

        indata = self.indata
        answers = self.answers

        nlayer = self.nlayer
        ncores = self.ncores
        activate = self.activate
        activate_output = self.activate_output

        alpha = self.alpha
        layers = self.layers

        if out: start = time.clock()

        # forward steps
        if nlayer > 1:
            core_style = self.ncore_style[1]
            # print(core_style)

        Nbatches = n//batchsize
        if n % batchsize > 0:
            Nbatches += 1

        # one iter still corresponds to the full training set, multiple batch updates
        for i_iter in range(niter):
            aveerror = maxerror = 0.0
            if self.lr_sched:
                alpha = self.lr_sched_function(alpha,i_epoch)

            for b in range(Nbatches):
                # initialize update matrices for the NN layers
                dWs = [None for i in range(nlayer)]
                for ilayer in range(nlayer):
                    dWs[ilayer] = ncp.zeros((self.ncores[ilayer].nrow, self.ncores[ilayer].ncol))

                # Reduce batch size for final batch if needed
                if b == Nbatches-1 and n % batchsize > 0:
                    Bsize = n % batchsize
                else:
                    Bsize = batchsize

                n_start = b*Bsize
                n_end = n_start + Bsize

                # Do forward prop, backward prop, and calculate the desired update for each example in the batch
                for one in range(n_start,n_end):
                    input_vecs = nlayer * [0]  # list of input vectors at each level
                    matvec_results = nlayer * [0]  # list of matvec results at each level
                    output_vecs = nlayer * [0]  # list of output vectors at each level
                    error_vecs = nlayer * [0]  # list of error vectors at each level

                    # forward steps
                    # save vectors before they get changed by A/D or D/A

                    for m in range(nlayer):
                        if m == 0:
                            if n == self.ndata:
                                input_vec = indata[one].copy()
                            else:
                                ind = n_vec[one]
                                input_vec = indata[ind].copy()
                            #input_vec = ex.copy()
                        else:
                            input_vec = output_vecs[m - 1].copy()
                        if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                            if self.useGPU:
                                ivec0 = cp.ones(input_vec.size+1,dtype=input_vec.dtype)
                                ivec0[:-1] = input_vec.flatten()
                                input_vec = ivec0
                            else:
                                input_vec = ncp.append(input_vec, 1)
                        matvec_result = ncores[m].matvec(input_vec, debug=debug)

                        if m != (nlayer - 1):
                            output_vec = activate.apply(matvec_result)
                        else:
                            output_vec = activate_output.apply(matvec_result)

                        input_vecs[m] = input_vec
                        matvec_results[m] = matvec_result
                        output_vecs[m] = output_vec

                    # Network output
                    z = output_vecs[-1].copy()

                    # Calculate loss and derivative of loss
                    if layers[-1][2] > 1:
                        # Answer
                        y = ncp.zeros(len(z))
                        if n == self.ndata:
                            y[int(answers[one])]=1
                        else:
                            y[int(answers[ind])]=1

                    elif layers[-1][2] == 1:
                        # Special case of one output neuron
                        if n == self.ndata:
                            y = int(answers[one])
                        else:
                            y = int(answers[ind])

                    L1_err = y - z
                    if self.costFunction == "L2":
                        error = 0.5*(L1_err)**2
                        error_vecs[nlayer-1] = L1_err
                    elif self.costFunction == "cross_entropy":
                        epsilon = 1e-12 # prevent zero argument in logarithm or division
                        error = -(y*ncp.log(z+epsilon) + (1-y)*ncp.log(1-z+epsilon))
                        error_vecs[nlayer-1] = L1_err/(z*(1-z)+epsilon)

                    # backward steps

                    for m in range(nlayer - 1, -1, -1):
                        ### Do update, update = p*input_vec*alpha
                            # p = next layer backpropagated error (err) * derivative of activation fxn (dvec)
                            # input_vec = output of previous layer
                            # alpha = learning rate

                        if m == (nlayer-1) and self.costFunction == "cross_entropy" and \
                        (self.activate_output.style == SIGMOID or self.activate_output.style == SOFTMAX):
                            p = L1_err
                        else:
                            dvec = activate.derivative(matvec_results[m])
                            p = error_vecs[m] * dvec

                        if m:
                            input_vec = output_vecs[m - 1].copy()
                            if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                                if self.useGPU:
                                    ivec0 = cp.ones(input_vec.size+1,dtype=input_vec.dtype)
                                    ivec0[:-1] = input_vec.flatten()
                                    input_vec = ivec0
                                else:
                                    input_vec = ncp.append(input_vec, 1)
                        else:
                            input_vec = input_vecs[0].copy()

                        # Compute the update for this layer
                        dW_m = ncp.outer(p, input_vec)

                        # Digitally accumulate weight updates
                        dWs[m] += dW_m

                        ## calculate error to send backwards
                        if m:
                            # self.plist.append(p)
                            error_vecs[m - 1] = ncores[m].vecmat(p, debug=debug)
                            if self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias":
                                error_vecs[m - 1] = error_vecs[m - 1][:-1]

                                # COMMENT IN/OUT: stats for size of actual update
                                # POS = same value as in Ncore

                                # POS = 1
                                # uactual = ncores[m].matrix - ucopy
                                # if ncores[m].wtmodel == POS: uactual /= ncores[m].alpha
                                # if one<n_updates:
                                #   self.uactuals[m][one] = uactual

                    # update statistics on error
                    error = ncp.linalg.norm(error, 2)
                    aveerror += error
                    maxerror = max(maxerror, error)

                # Perform update at end of batch
                for m in range(nlayer):
                    ncores[m].update_burst(dWs[m], alpha, debug=debug)

            aveerror /= n
            if out: print(i_iter, aveerror, maxerror)

        if out:
            stop = time.clock()
            print("time =", stop - start, "secs");

        if minmax:
            for m in range(nlayer):
                wtmax = np.amax(ncores[m].get_matrix)
                wtmin = np.amin(ncores[m].get_matrix)
                print("Matrix %d: max1 %g min1 %g max2 %g min2 %g" % \
                      (wtmax, wtmin, m + 1))

        # Update learning rate
        self.alpha = alpha

        return aveerror, maxerror  # ,ncores[0].matrix,ncores[1].matrix


    def train_simd(self, n=0, niter=1, out=0, minmax=0, batchsize=10, i_epoch=0, debug=0):
        """

        Accelerate training using SIMD techniques, i.e. a single MVM/OPU in CrossSim executes multiple 
        xbar reads/writes. The number of examples processed in parallel is the batchsize, so this
        implements minibatch gradient descent

        The accelearation is much more significant on a GPU. Training can be accelerated by at least 
        2.5x without parasitics and 1.5x with parasitics (no LUT), for an appropriately chosen batchsize

        This method is called from train() after checking whether SIMD is requested in all the layer params

        stored matrices are adjusted as training proceeds
        n = # of inputs to train on, 0 = default = all
        niter = # of training iterations to perform
        out = 1 for timing output
        minmax = 1 for output of matrix bounds
        """

        if not self.ncore_style: error("Neural core is not initialized")
        if not n: n = self.ndata
        if n > self.ndata: error("N too large for training")
        
        if n != self.ndata:
            n_vec = np.random.uniform(0,self.ndata-1,n)
            n_vec = np.round(n_vec).astype(int)

        indata = self.indata
        answers = self.answers

        nlayer = self.nlayer
        ncores = self.ncores
        activate = self.activate
        activate_output = self.activate_output

        alpha = self.alpha
        layers = self.layers

        if out: start = time.clock()

        # Expand the weight matrices for SIMD
        for ilayer in range(nlayer):
            self.ncores[ilayer].neural_core.expand_matrix(batchsize,mode=1)

        # forward steps
        if nlayer > 1:
            core_style = self.ncore_style[1]
            # print(core_style)

        Nbatches = n//batchsize
        if n % batchsize > 0:
            # May support spillover later on, but having multiple Nex_par values will reduce efficiency
            raise ValueError("If using SIMD, training set size must be divisible by batch size")

        for i_iter in range(niter):
            aveerror = maxerror = 0.0
            if self.lr_sched:
                alpha = self.lr_sched_function(alpha,i_epoch)

            for b in range(Nbatches):

                if b == 0 or (b+1) % 1000 == 0:
                    print("Batch "+str(b+1)+"/"+str(Nbatches))

                n_start = b*batchsize
                n_end = n_start + batchsize

                input_vecs = nlayer * [0]  # list of input vectors at each level
                matvec_results = nlayer * [0]  # list of matvec results at each level
                output_vecs = nlayer * [0]  # list of output vectors at each level
                error_vecs = nlayer * [0]  # list of error vectors at each level

                for m in range(nlayer):
                    add_bias = (self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias")
                    if m == 0:
                        # Pull out inputs from training set and concatenate into a long vector
                        ilen = len(indata[0])
                        if add_bias: ilen += 1
                        input_vec = np.zeros(ilen*batchsize)
                        k = 0
                        for one in range(n_start,n_end):
                            if n == self.ndata:
                                ivec = indata[one].copy()
                            else:
                                ind = n_vec[one]
                                ivec = indata[ind].copy()  

                            if add_bias: ivec = np.append(ivec, 1)

                            input_vec[k*ilen:((k+1)*ilen)] = ivec.copy()
                            k += 1
                    else:
                        ivec = output_vecs[m-1].copy()
                        # Add bias elements
                        if add_bias:
                            ilen0 = len(ivec)//batchsize
                            ilen1 = ilen0 + 1
                            input_vec = np.zeros(ilen1*batchsize)
                            for k in range(batchsize):
                                input_vec[k*ilen1:((k+1)*ilen1-1)] = ivec[k*ilen0:((k+1)*ilen0)]
                                input_vec[(k+1)*ilen1-1] = 1
                        else:
                            input_vec = ivec

                    matvec_result = ncores[m].matvec(input_vec, debug=debug)

                    if m != (nlayer - 1):
                        output_vec = activate.apply(matvec_result)
                    else:
                        if activate_output.style == SOFTMAX:
                            # Apply softmax to one output at a time
                            mlen = len(matvec_result)//batchsize
                            output_vec = np.zeros(mlen*batchsize)
                            for k in range(batchsize):
                                output_vec[k*mlen:(k+1)*mlen] = activate_output.apply(matvec_result[k*mlen:(k+1)*mlen])
                        else:
                            output_vec = activate_output.apply(matvec_result)

                    input_vecs[m] = input_vec
                    matvec_results[m] = matvec_result
                    output_vecs[m] = output_vec

                # Separate the outputs for individual examples and calculate the error
                output_length = layers[-1][2]
                error_vecs[nlayer-1] = np.zeros(batchsize*output_length)
                L1_err = np.zeros(batchsize*output_length)
                error = np.zeros(batchsize*output_length)

                k = 0
                for one in range(n_start,n_end):
                    output_k = output_vecs[-1][k*output_length:((k+1)*output_length)]
                    # Network output
                    z = output_k.copy()
                    # Assume there is more than one output neuron
                    y = np.zeros(output_length)
                    if n == self.ndata:
                        y[int(answers[one])] = 1
                    else:
                        y[int(answers[n_vec[one]])] = 1

                    L1_err[k*output_length:((k+1)*output_length)] = y - z
                    if self.costFunction == "L2":
                        error[k*output_length:((k+1)*output_length)] = 0.5*(y-z)**2
                        error_vecs[nlayer-1][k*output_length:((k+1)*output_length)] = y-z
                    elif self.costFunction == "cross_entropy":
                        epsilon = 1e-12 # prevent zero argument in logarithm or division
                        error[k*output_length:((k+1)*output_length)] = -(y*np.log(z+epsilon) + (1-y)*np.log(1-z+epsilon))
                        error_vecs[nlayer-1][k*output_length:((k+1)*output_length)] = (y-z)/(z*(1-z)+epsilon)
                    k += 1

                # backward steps
                for m in range(nlayer - 1, -1, -1):

                    if m == (nlayer-1) and self.costFunction == "cross_entropy" and \
                        (self.activate_output.style == SIGMOID or self.activate_output.style == SOFTMAX):
                        p = L1_err
                    else:
                        dvec = activate.derivative(matvec_results[m])
                        p = error_vecs[m] * dvec

                    if m:
                        input_vec = input_vecs[m].copy()
                    else:
                        input_vec = input_vecs[0].copy()

                    # Compute the update for this layer
                    ncores[m].update(p, input_vec, alpha, debug=debug)

                    ## calculate error to send backwards
                    if m:
                        # self.plist.append(p)
                        error_vecs[m - 1] = ncores[m].vecmat(p, debug=debug)

                        add_bias = (self.ncore_style[m] == "old_bias" or self.ncore_style[m] == "new_bias")
                        # Remove bias element from VMM result
                        if add_bias:
                            evec = error_vecs[m - 1].copy()
                            elen0 = len(evec)//batchsize
                            elen1 = elen0 - 1
                            evec_new = np.zeros(elen1*batchsize)
                            for k in range(batchsize):
                                evec_new[k*elen1:(k+1)*elen1] = evec[k*elen0:((k+1)*elen0-1)]
                            error_vecs[m - 1] = evec_new

                # update statistics on error
                error = 0
                for k in range(batchsize):
                    error += np.linalg.norm(error_vecs[-1][k*output_length:((k+1)*output_length)],2)
                error /= batchsize
                aveerror += error
                maxerror = max(maxerror, error)

            aveerror /= n
            if out: print(i_iter, aveerror, maxerror)

        if out:
            stop = time.clock()
            print("time =", stop - start, "secs");

        if minmax:
            for m in range(nlayer):
                wtmax = np.amax(ncores[m].get_matrix)
                wtmin = np.amin(ncores[m].get_matrix)
                print("Matrix %d: max1 %g min1 %g max2 %g min2 %g" % \
                      (wtmax, wtmin, m + 1))

        # Undo the matrix expansion
        for ilayer in range(self.nlayer):
            self.ncores[ilayer].neural_core.unexpand_matrix(mode=1)

        # Update learning rate
        self.alpha = alpha

        return aveerror, maxerror  # ,ncores[0].matrix,ncores[1].matrix


    def import_digital_bias(self,weight_dict,bias_bits,layerParams):

        for ilayer in range(self.nlayer):

            if self.layerTypes[ilayer] not in ("conv","dense"):
                continue

            # Extract raw tensors from Keras model
            Wi_0 = weight_dict[layerParams[ilayer]['name']]

            # Extract the bias weights
            if self.digital_bias[ilayer]:
                Wbias = Wi_0[1].astype(precision)

                # Quantize the bias values (set bias_bits = 0 to disable)
                # Range is set by the minimum and maximum
                # Options: for bias_bit,
                #   0       : no quanziation
                #   int > 0 : quantize to this number of bits with range set by min and max bias values
                #   "adc"   : set the range to the ADC range and the bias_bits to the adc resolution
                #       If a bigger range is needed than ADC, set the range to a power of 2 times the ADC range
                #       and add the appropriate number of bits
                if bias_bits != "adc":
                    if bias_bits > 0:
                        Bmax = np.max(Wbias)
                        Bmin = np.min(Wbias)
                        qmult = (2 ** bias_bits) / (Bmax - Bmin)
                        Wbias -= Bmin
                        Wbias *= qmult
                        Wbias = np.rint(Wbias, out=Wbias)
                        Wbias /= qmult
                        Wbias += Bmin

                else:
                    # Find the range of bias weights
                    Bmax = np.max(Wbias)
                    Bmin = np.min(Wbias)
                    
                    # Find ADC range of corresponding layer
                    adc_min = self.ncores[ilayer].params.algorithm_params.row_output.minimum
                    adc_max = self.ncores[ilayer].params.algorithm_params.row_output.maximum
                    adc_bits = self.ncores[ilayer].params.xbar_params.row_output.bits

                    if adc_bits > 0:

                        # If the layer has multiple cores, its range effectively expands when added
                        if self.ncores[ilayer].Ncores > 1:
                            expand_bits = np.ceil(np.log2(self.ncores[ilayer].Ncores))
                            adc_min *= pow(2,expand_bits)
                            adc_max *= pow(2,expand_bits)
                            adc_bits += expand_bits

                        if Bmax < adc_max and Bmin > adc_min:
                            # Bias range is contained inside ADC range
                            b_min = adc_min
                            b_max = adc_max
                            nbits = adc_bits

                        elif Bmax > adc_max and Bmin > adc_min:
                            # Bias max is larger than ADC max
                            extend_bits = np.ceil(np.log2((Bmax-adc_min)/(adc_max-adc_min)))
                            b_max = adc_min + (adc_max-adc_min) * pow(2,extend_bits)
                            b_min = adc_min
                            nbits = adc_bits + extend_bits

                        elif Bmax < adc_max and Bmin < adc_min:
                            # Bias min is smaller than ADC min
                            extend_bits = np.ceil(np.log2((adc_max-Bmin)/(adc_max-adc_min)))
                            b_max = adc_max
                            b_min = adc_max - (adc_max - adc_min) * pow(2,extend_bits)
                            nbits = adc_bits + extend_bits

                        elif Bmax > adc_max and Bmin < adc_min:
                            # Bias limits are beyond both limits of ADC
                            # First extend min, then extend max
                            extend_bits_min = np.ceil(np.log2((adc_max-Bmin)/(adc_max-adc_min)))
                            b_min = adc_max - (adc_max - adc_min) * pow(2,extend_bits_min)
                            extend_bits_max = np.ceil(np.log2((Bmax-b_min)/(adc_max-b_min)))
                            b_max = b_min + (adc_max-b_min) * pow(2,extend_bits_max)
                            nbits = adc_bits + extend_bits_min + extend_bits_max

                        qmult = (2 ** nbits) / (b_max - b_min)
                        Wbias -= b_min
                        Wbias *= qmult
                        Wbias = np.rint(Wbias, out=Wbias)
                        Wbias /= qmult
                        Wbias += b_min

                if self.layerTypes[ilayer] == "conv":
                    Wbias = Wbias.reshape((len(Wbias),1,1))
                if self.useGPU:
                    Wbias = cp.array(Wbias)
                self.bias_values[ilayer] = Wbias


    def update_neuron_statistics(self,m,ivec,mvec,mvec_all):

        # Collect all ADC input values (beware of memory consumption!)
        # This function is used when input or weight bit slices are not enabled

        # If layer has no activation, do not account for a ReLU during profiling
        if m != self.nlayer - 1:
            noRelu = (self.activations[m].style != RECTLINEAR)
        else:
            noRelu = (self.activate_output.style != RECTLINEAR)

        if self.profile_ADC_inputs:
            # Single core
            if mvec_all is None:
                if not self.profile_ADC_biased or (self.profile_ADC_biased and noRelu):
                    ADC_inputs = ncp.array(mvec.flatten(),dtype=ncp.float32)
                    self.profiled_values[m] = ncp.concatenate((self.profiled_values[m],ADC_inputs))

                elif self.profile_ADC_biased:
                    ADC_inputs = mvec[mvec > -self.bias_values[m]].flatten()
                    ADC_inputs = ncp.array(ADC_inputs,dtype=ncp.float32)
                    self.profiled_values[m] = ncp.concatenate((self.profiled_values[m],ADC_inputs))

            # Multi core profiling: ignore profile_ADC_biased setting
            else:
                all_cores_values = ncp.zeros(0,dtype=ncp.float32)
                for k in range(len(mvec_all)):
                    ADC_inputs = ncp.array(mvec_all[k].flatten(),dtype=ncp.float32)
                    all_cores_values = ncp.concatenate((all_cores_values,ADC_inputs))
                self.profiled_values[m] = ncp.concatenate((self.profiled_values[m],all_cores_values))

        elif self.profile_DAC_inputs:
            self.profiled_values[m] = ncp.concatenate((self.profiled_values[m],ncp.array(ivec.flatten(),dtype=ncp.float32)))

    # Used for Nvidia INT4 model only
    def import_quantization(self,weight_dict,layerParams):

        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] != "quantize":
                continue
            # Extract scale factors from Keras model
            W_q = weight_dict[layerParams[ilayer]['name']][0]
            W_q = W_q.reshape((len(W_q),1,1))
            if self.useGPU:
                W_q = cp.array(W_q)
            self.quantization_values[ilayer] = W_q

    # Used for Nvidia INT4 model only
    def import_scale(self,weight_dict,layerParams):

        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] != "scale":
                continue
            # Extract scale factors from Keras model
            W_s = weight_dict[layerParams[ilayer]['name']][0]
            if self.useGPU:
                W_s = cp.array(W_s)
            self.scale_values[ilayer] = W_s