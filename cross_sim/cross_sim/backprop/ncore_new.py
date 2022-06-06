#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""
All crossbar model settings must be set by setting the params object

"""

from warnings import warn

import numpy as np

from ..xbar_simulator import MakeCore, MakeMultiCore
from ..xbar_simulator import Parameters


def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# Ncore class

class NcoreXbar:
    """
    This class models a neural core using the xbar_simulator code.
    It has 3 kernel methods invoked on
    the crossbar by calling algorithms:
      matvec multiply, vecmat multiply, outer-product update

    """

    def __init__(self, nrow, ncol, precision=np.float32, **kwargs):
        """
        Instantiate neural core and set core params
        For convolution, nrow and ncol are not required (set to 0)
        For fully connected, nrow and ncol are needed
        params = parameter settings for neural cores
        
        Split MVMs are handled:
            - Inside ConvolutionCore if core type is convolution: still use a single neural_core here
            - Inside NcoreXbar (this file) if fully connected as a list of neural_core objects
                Note: only set_matrix(), get_matrix(), and matvec() currently support
                        split MVM functionality

         """

        if "params" not in kwargs:
            raise ValueError("params must be passed as an argument when use_params_only is set")
        if "clip" in kwargs or "clipmin" in kwargs or "clipmax" in kwargs or "truncate" in kwargs or "wtmodel"in kwargs or "inner" in kwargs:
            warn("Neural core settings ignored, only passed params used")
        params = kwargs["params"]

        # ****** create a neural core

        # Set master params object
        if type(params) is not list:
            self.params = params
        else:
            self.params = params[0].copy()

        # Determine number of cores
        NrowsMax = self.params.xbar_params.NrowsMax
        if self.params.convolution_parameters.is_conv_core:
            self.params.convolution_parameters.post_set()
            ncol = self.params.convolution_parameters.Nrows
            nrow = self.params.convolution_parameters.Noc
        if NrowsMax > 0:
            self.Ncores = (ncol-1)//NrowsMax + 1
        else:
            self.Ncores = 1

        # Check that Ncores is compatible with the number of params objects
        if self.Ncores == 1 and type(params) is list:
            raise ValueError("Too many params objects provided for single-core layer")
        if self.Ncores > 1 and type(params) is not list:
            raise ValueError("Must provide a list of params objects for multi-core")
        if self.Ncores > 1 and type(params) is list and len(params) != self.Ncores:
            raise ValueError("Number of params objects provided does not match number of cores")

        # Create core
        if self.Ncores == 1:
            self.neural_core = MakeCore(params=self.params)
        else:
            if self.params.convolution_parameters.is_conv_core:
                self.neural_core = MakeMultiCore(params=params)

            else:
                self.neural_cores = [None for i in range(self.Ncores)]

                # Create sub-cores
                for i in range(self.Ncores):
                    self.neural_cores[i] = MakeCore(params=params[i])

                # NrowsVec is a list of the number of rows in the sub-cores
                # Split the rows up as evenly as possible among the arrays
                if ncol % self.Ncores == 0:
                    self.NrowsVec = (ncol // self.Ncores)*np.ones(self.Ncores)
                else:
                    Nrows1 = np.round(ncol / self.Ncores)
                    self.NrowsVec = np.zeros(self.Ncores)
                    for k in range(self.Ncores-1):
                        self.NrowsVec[k] = Nrows1
                    self.NrowsVec[self.Ncores-1] = ncol - (self.Ncores-1)*Nrows1
                self.NrowsVec = self.NrowsVec.astype(int)
            
        # Initialize matrix to zero
        self.nrow = nrow
        self.ncol = ncol
        
        global ncp
        if self.params.numeric_params.useGPU:
            global cp
            import cupy as cp
            cp.cuda.Device(self.params.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np


    # -------------------------------------------------------
    # Allow caller to set and get matrix
    # caller uses POSNEG values, Ncore may store them differently
    # -------------------------------------------------------

    def set_matrix(self, matrix, applyErrors=True, verbose=False):
        """
        store matrix after initialized by caller
        remap weights if required by wtmodel
        """
        if verbose: print("Min/Max matrix values", np.min(matrix), np.max(matrix));

        if self.Ncores == 1 or self.params.convolution_parameters.is_conv_core:
            self.neural_core.set_matrix(matrix, applyErrors=applyErrors)
        else:
            for k in range(self.Ncores):
                i_start = np.sum(self.NrowsVec[:k])
                i_end = np.sum(self.NrowsVec[:k+1])
                matrix_k = matrix[:,i_start:i_end]
                self.neural_cores[k].set_matrix(matrix_k, applyErrors=applyErrors)


    def get_matrix(self):
        """
        return matrix to caller
        unmap weights if required by wtmodel
        """

        if self.Ncores == 1 or self.params.convolution_parameters.is_conv_core:
            return self.neural_core._read_matrix()
        else:
            matrix = ncp.zeros((nrow,ncol))
            for k in range(self.Ncores):
                i_start = np.sum(self.NrowsVec[:k])
                i_end = np.sum(self.NrowsVec[:k+1])
                matrix[:,i_start:i_end] = self.neural_cores[k]._read_matrix()
            return matrix


    # -------------------------------------------------------
    # matrix-vector multiply kernel
    # -------------------------------------------------------

    def matvec(self, vec, **kwargs):
        """
        perform matrix-vector multiply on analog vector
        apply readnoise model to matrix values
        d2a and a2d conversions on in/out vectors
        does NOT modify input vector
        if weights are POS, adjust result as if weights had been POSNEG
        """

        if self.Ncores == 1 or self.params.convolution_parameters.is_conv_core:
            return self.neural_core.run_xbar_mvm(vec)

        else:
            profiling = kwargs.get("profiling",False)
            if not profiling:
                vec_0 = vec[:self.NrowsVec[0]]
                M0 = self.neural_cores[0].run_xbar_mvm(vec_0)
                output = M0.copy()
                for k in range(1,self.Ncores):
                    i_start = np.sum(self.NrowsVec[:k])
                    i_end = np.sum(self.NrowsVec[:k+1])
                    vec_k = vec[i_start:i_end]
                    Mk = self.neural_cores[k].run_xbar_mvm(vec_k)
                    output += Mk.copy()
                return output
            else:
                output_all = [None]*self.Ncores
                vec_0 = vec[:self.NrowsVec[0]]
                M0 = self.neural_cores[0].run_xbar_mvm(vec_0)
                output = M0
                output_all[0] = M0.copy()
                for k in range(1,self.Ncores):
                    i_start = np.sum(self.NrowsVec[:k])
                    i_end = np.sum(self.NrowsVec[:k+1])
                    vec_k = vec[i_start:i_end]
                    Mk = self.neural_cores[k].run_xbar_mvm(vec_k)
                    output_all[k] = Mk.copy()
                    output += Mk
                return output, output_all

    # -------------------------------------------------------
    # vector-matrix multiply kernel
    # -------------------------------------------------------

    def vecmat(self, vec, **kwargs):
        """
        perform vector-matrix multiply on analog vector
        """
        return self.neural_core.run_xbar_vmm(vec)


    # -------------------------------------------------------
    # outer-product matrix update kernel
    # -------------------------------------------------------

    def update(self, vec1, vec2, rate, **kwargs):
        """
        perform matrix update via outer product of 2 analog vectors
        """

        self.neural_core.update_matrix(vec1, vec2, learning_rate=rate)


    # -------------------------------------------------------
    # serial matrix update kernel
    # -------------------------------------------------------

    def update_burst(self, mat, rate, **kwargs):
        """
        perform matrix update via outer product of 2 analog vectors
        """

        self.neural_core.update_matrix_burst(mat, learning_rate=rate)

    # -------------------------------------------------------
    # CONV only: reshape input vector to shape appropriate for conv layer
    # -------------------------------------------------------

    def reshape_input(self,M_input):
        if self.params.convolution_parameters.is_conv_core:
            return self.neural_core.reshape_input(M_input)
        else:
            raise ValueError('Invalid method on non-convolutional core')


    # -------------------------------------------------------
    # CONV only: apply convolution
    # -------------------------------------------------------

    def apply_convolution(self,M_input,profiling=False):
        if self.params.convolution_parameters.is_conv_core:
            return self.neural_core.apply_convolution(M_input,profiling)
        else:
            raise ValueError('Invalid method on non-convolutional core')


    # -------------------------------------------------------
    # CONV only: apply max pool (if specified in layer)
    # -------------------------------------------------------

    def max_pool(self,M_input):
        if self.params.convolution_parameters.is_conv_core:
            return self.neural_core.max_pool(M_input)
        else:
            raise ValueError('Invalid method on non-convolutional core')


    # -------------------------------------------------------
    # CONV only: apply convolution and max pool (if specified in layer)
    # -------------------------------------------------------

    def convolve_and_pool(self,M_input):
        if self.params.convolution_parameters.is_conv_core:
            return self.neural_core.convolve_and_pool(M_input)
        else:
            raise ValueError('Invalid method on non-convolutional core')


    # -------------------------------------------------------
    # CONV only: flatten output so that it can be received by FC layer
    # -------------------------------------------------------

    def flatten_layer(self,M_input):
        if self.params.convolution_parameters.is_conv_core:
            return self.neural_core.flatten_layer(M_input)
        else:
            raise ValueError('Invalid method on non-convolutional core')

    # -------------------------------------------------------
    # return a vector of target and real updates for this core
    # -------------------------------------------------------

    def get_update_record(self):
        # Only call this if using NumericCore for the inner core
        return self.neural_core.get_update_record()


##################################  OLD CODE
##################################
    # -------------------------------------------------------
    # Define readnoise, writenoise, nonlinear, A2D/D2A models
    # pass additional params for weight model if needed
    # -------------------------------------------------------

    def reinitialize_neural_core(self):
        """
        This function re-creates the neural core.  to be used after changing any parameter settings to save the new parameters
        :return:
        """
        # first save matrix (using internal representation)
        internal_matrix = self.neural_core._save_matrix()
        # create new neural core
        self.neural_core = MakeCore(params=self.params, inner=self.inner, outer=self.outer)
        # restore matrix
        self.neural_core._restore_matrix(internal_matrix)

    def set_readnoise(self, **kwargs):
        """
        read noise models
        required args:
        style = GAUSS
          GAUSS = Gaussian spread of weight change
        sigma  = size of spread

        optional args:

        scale = FIXED,LOHI (def = FIXED)
          FIXED = delta is independent of weight
          LOHI = delta scales with weight from lo to hi across range

        truncate = 1/0 for whether to clip updated wt at range boundaries (def = 1)

        """
        if "style" not in kwargs: error("Style setting required")
        if "sigma" not in kwargs: error("Sigma setting required")
        if kwargs["style"] != "GAUSS":
            warn("read noise style "+str(kwargs["style"])+
                                      " not implemented for xbar_simulator, using GAUSS style")

        scale = kwargs.get("scale","FIXED")

        # check for allowflip setting and warn if it is not 1
        if kwargs.get("allowflip",1) !=1:
            warn ("flipping the midpoint is always allowed, ignoring allowflip")

        # set sigma
        self.params.numeric_params.read_noise.sigma = kwargs["sigma"]

        # map parameter truncate to read_noise.keep_within_range
        self.params.numeric_params.read_noise.keep_within_range = bool(kwargs.get("truncate",1))


        # use different read noise scaling models

        if scale == "FIXED":
            self.params.numeric_params.read_noise.proportional = False
        elif scale == "LOHI":
            #new value = old value + sigma*old value*weight_range/max_weight
            self.params.numeric_params.read_noise.proportional = True
            self.params.numeric_params.read_noise.sigma = kwargs["sigma"]* \
                              self.params.xbar_params.weights.range/self.params.xbar_params.weights.maximum
        else: # use fixed scaling and warn if scaling model not implemented
            self.params.numeric_params.read_noise.proportional = False
            warn("read noise scaling model model "+str(scale)+
                                      " not implemented for xbar_simulator, using FIXED scaling")

        #***** re-instantiate the neural core  (this code could be called once by the user instead of after every setting)
        self.reinitialize_neural_core()


    def set_writenoise(self, **kwargs):
        """
          write noise models
          required args:
            style = GAUSS
            sigma = size of spread
          optional args:

            scale = SQRTFIXED,SQRTLOHI,SQRTHILO (def = FIXED)
              SQRTFIXED = noise scales with sqrt(delta), independent of weight
              SQRTLOHI = noise scales with sqrt(delta), prop to weight from lo to hi
              SQRTHILO = noise scales with sqrt(delta), prop to weight from hi to lo

        """

        if "sigma" not in kwargs: error("Sigma setting required")

        if kwargs.get("style","GAUSS") != "GAUSS":
            warn("read noise model "+str(kwargs["style"])+
                                      " not implemented for xbar_simulator, using GAUSS style")

        # check for allowflip setting and warn if it is not 1
        if kwargs.get("allowflip",1) !=1:
            warn ("flipping the midpoint is always allowed, ignoring allowflip")

        scale = kwargs.get("scale","FIXED")
        # optional args

        # set sigma
        self.params.numeric_params.write_noise.sigma = kwargs["sigma"]

        # set the scale type:
        if scale == "SQRTFIXED":
            self.params.numeric_params.write_noise.write_noise_model = "G_INDEPENDENT"
        elif scale == "SQRTLOHI":
            self.params.numeric_params.write_noise.write_noise_model = "G_PROPORTIONAL"
        elif scale == "SQRTHILO":
            self.params.numeric_params.write_noise.write_noise_model = "G_INVERSE"
        else:
            warn("write noise scaling model model "+str(scale)+
                          " not implemented for xbar_simulator, using SQRTFIXED scaling")
            self.params.numeric_params.write_noise.write_noise_model = "G_INDEPENDENT"

        #***** re-instantiate the neural core  (this code could be called once by the user instead of after every setting)
        self.reinitialize_neural_core()




    def set_nonlinear(self, **kwargs):
        """
        style = ASYM,ASYMLOOP,SYM
        alpha = degree of nonlinearity, smaller is more linear, 0.0 = linear
        :return:
        """
        if "style" not in kwargs: error("Style setting required")
        if "alpha" not in kwargs: error("Alpha setting required")
        style = kwargs["style"]

        self.params.numeric_params.nonlinearity.alpha = kwargs["alpha"]

        if style == "SYM":
            self.params.numeric_params.nonlinearity.symmetric = True
        elif style == "ASYM":
            self.params.numeric_params.nonlinearity.symmetric = False
        else:
            warn("nonlinearlity style "+str(kwargs["style"])+
                                      " not implemented for xbar_simulator, using ASYM style")
            self.params.numeric_params.nonlinearity.symmetric = False

        # ***** re-instantiate the neural core  (this code could be called once by the user instead of after every setting)
        self.reinitialize_neural_core()



    def set_a2d(self, **kwargs):
        raise NotImplementedError("a2d settings model not implemented")
        # self.a2d = A2D(**kwargs)
        # if self.a2d.inactive: self.a2d = None

    def set_d2a(self, **kwargs):
        raise NotImplementedError("d2a settings model not implemented")
        # self.d2a = D2A(**kwargs)
        # if self.d2a.inactive: self.d2a = None
