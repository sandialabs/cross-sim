#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__authors__ = 'txiao'

from .icore import ICore
from ..parameters import Parameters
import numpy as np
from ..cores import  WrapperCore
from .core_initialization import MakeCore2, verify_parameters
from ..parameters.parameter_defaults import ZeroResetPCEnum, SimTypeEnum, CrossbarTypeEnum

class ConvolutionCore(ICore):
    '''
    Implements a crossbar that performs a convolutional kernel
    ---- INFERENCE ONLY! ----
    Outer product update and VMM are not implemented for convolutions
    '''

    def __init__(self, params, Ncores=1):
        """

        :param params: All parameters
        :type params: Parameters
        :return:
        """
        # Set the params of the Convolution core
        # If the Convolution core contains multiple neural cores, the master params object is a copy of the first
        # subcore params
        if type(params) is not list:
            self.params = params
        else:
            self.params = params[0].copy()

        # Create a core
        # This needs to be called even if multi-core to ensure the ConvolutionCore parameters
        # are appropriately post set
        self.core = MakeCore2(self.params)

        self.Ncores = Ncores

        if self.Ncores > 1:
            self.core = None
            self.cores = [None]*Ncores
            self.NrowsMax = self.params.xbar_params.NrowsMax

            # Split the rows up as evenly as possible among the arrays
            # Rules:
            #   If ncol is a multiple of NrowsMax, split evenly
            #   If ncol is a multiple of NrowsMax/2 or NrowsMax/4, split so that all but one core has NrowsMax
            #       This condition exists so that convolution blocks in depthwise convolutions are not split.
            #       It does not catch all edge cases
            #   If none of the above, split evenly
            if self.params.convolution_parameters.Nrows % self.Ncores == 0:
                self.NrowsVec = (self.params.convolution_parameters.Nrows // self.Ncores)*np.ones(self.Ncores)
            else:
                if (self.params.convolution_parameters.Nrows % (self.NrowsMax/2) == 0) or \
                    (self.params.convolution_parameters.Nrows % (self.NrowsMax/4) == 0):
                    self.NrowsVec = np.zeros(self.Ncores)
                    for k in range(self.Ncores-1):
                        self.NrowsVec[k] = self.NrowsMax
                    self.NrowsVec[self.Ncores-1] = self.params.convolution_parameters.Nrows - (Ncores-1)*self.NrowsMax
                else:
                    Nrows1 = np.round(self.params.convolution_parameters.Nrows / self.Ncores)
                    self.NrowsVec = np.zeros(self.Ncores)
                    for k in range(self.Ncores-1):
                        self.NrowsVec[k] = Nrows1
                    self.NrowsVec[self.Ncores-1] = self.params.convolution_parameters.Nrows - (Ncores-1)*Nrows1
            self.NrowsVec = self.NrowsVec.astype(int)

            for k in range(Ncores):
                if type(params) is list:
                    params_k = params[k]
                else:
                    print("Individual params not provided for subcores of a layer: copying the same params onto all subcores")
                    params_k = self.params.copy()
                params_k.convolution_parameters.Nrows = self.NrowsVec[k]
                params_k.convolution_parameters.subarray_id = k
                params_k.convolution_parameters.last_subarray = (k == Ncores - 1)
                self.cores[k] = MakeCore2(params_k)

        global ncp
        if self.params.numeric_params.useGPU:
            global cp
            import cupy as cp
            cp.cuda.Device(0).use()
            ncp = cp
        else:
            ncp = np


    def set_matrix(self, matrix, applyErrors=True):
        '''
        Set the weight matrix across all constituent wrapper cores
        The crossbar arrangement is as follows:
        - Along the rows are ALL the kernel weights for each input channel. Input channel 0 first, then input channel 1, ...
        - Along the columns are the weights for the different output channels
        '''
        Kx = self.params.convolution_parameters.Kx
        Ky = self.params.convolution_parameters.Ky
        Nic = self.params.convolution_parameters.Nic
        Noc = self.params.convolution_parameters.Noc
        Nrows = self.params.convolution_parameters.Nrows
        Nox = self.params.convolution_parameters.Nox
        Noy = self.params.convolution_parameters.Noy
        x_par = self.params.numeric_params.x_par
        y_par = self.params.numeric_params.y_par

        if ((Nox % x_par) != 0) or ((Noy % y_par) != 0):
            print('Warning: # sliding windows in a block ('+str(x_par)+','+str(y_par)+') ' \
                +'not divisible by total # windows ('+str(Nox)+','+str(Noy)+')')
        if x_par > Nox or y_par > Noy:
            raise ValueError('# sliding windows in a block ('+str(x_par)+','+str(y_par)+') ' \
                +'cannot be larger than output feature dimensions ('+str(Nox)+','+str(Noy)+')')

        # Check number of rows
        # matrix.shape[0] for VMM, matrix.shape[1] for MVM
        if matrix.shape[1] != Nrows:
            raise ValueError("The number of rows in the weight matrix is inconsistent with conv core parameters")

        # Check number of columns
        # matrix.shape[1] for VMM, matrix.shape[0] for MVM
        if matrix.shape[0] != Noc:
            raise ValueError("The number of columns in the weight matrix is inconsistent with conv core parameters")

        if self.Ncores == 1:
            self.core.set_matrix(matrix, applyErrors=applyErrors)

        else:
            NrowsMax = self.NrowsMax
            for k in range(self.Ncores):
                i_start = np.sum(self.NrowsVec[:k])
                i_end = np.sum(self.NrowsVec[:k+1])
                matrix_k = matrix[:,i_start:i_end]
                self.cores[k].set_matrix(matrix_k, applyErrors=applyErrors)


    def set_vmm_inputs(self, vector):
        '''
        Set the vector to be used in a VMM operation
        '''
        if self.Ncores == 1:
            return self.core.set_vmm_inputs(vector)
        else:
            raise NotImplementedError("Backpropagation through a split matrix not yet supported")


    def set_mvm_inputs(self, vector):
        '''
        Set the vector to be used in a MVM operation
        '''
        if self.Ncores == 1:
            return self.core.set_mvm_inputs(vector)
        else:
            Nrows = self.params.convolution_parameters.Nrows
            Ncopy = self.params.numeric_params.x_par*self.params.numeric_params.y_par
            vector = vector.reshape((Ncopy,Nrows))
            for k in range(self.Ncores):
                i_start = np.sum(self.NrowsVec[:k])
                i_end = np.sum(self.NrowsVec[:k+1])
                vector_k = vector[:,i_start:i_end].flatten()
                self.cores[k].set_mvm_inputs(vector_k)

    
    def run_xbar_vmm(self,vector=None):
        '''
        Run VMM on the core
        '''
        if vector is not None:
            self.set_vmm_inputs(vector)

        if self.Ncores == 1:
            return self.core.run_xbar_vmm()
        else:
            raise NotImplementedError("Backpropagation through a split matrix not yet supported")

    def run_xbar_mvm(self, vector=None, profiling=False):
        '''
        Run MVM on the core: this returns the result of applying the conv kernel to one sliding window, across all input ...
        channels, for all output channels
        Result is a vector of dimension Noc x 1
        '''
        if vector is not None:
            self.set_mvm_inputs(vector)

        if self.Ncores == 1:
            return self.core.run_xbar_mvm()
        else:
            if not profiling:
                output = self.cores[0].run_xbar_mvm()
                for k in range(1,self.Ncores):
                    output += self.cores[k].run_xbar_mvm()
                return output
            else:
                output_all = [None]*self.Ncores
                M0 = self.cores[0].run_xbar_mvm()
                output = M0
                output_all[0] = M0.copy()
                for k in range(1,self.Ncores):
                    Mk = self.cores[k].run_xbar_mvm()
                    output_all[k] = Mk.copy()
                    output += Mk

                return output, output_all

    def update_matrix(self, row_vector, col_vector, learning_rate=1):
        r'''
        Updates the matrix given input row and column vectors.
        assumes serial updates are done by col (one col at a time)

        '''
        if self.Ncores == 1:
            self.core.update_matrix(row_vector,col_vector,learning_rate)
        else:
            raise NotImplementedError("Backpropagation through a split matrix not yet supported")


    def _read_matrix(self):
        '''
        Read the internal matrix held by this core
        '''
        if self.Ncores == 1:
            return self.core._read_matrix()
        else:
            matrix = ncp.zeros((self.params.convolution_parameters.Noc,self.params.convolution_parameters.Nrows))
            for k in range(self.Ncores):
                i_start = np.sum(self.NrowsVec[:k])
                i_end = np.sum(self.NrowsVec[:(k+1)])
                matrix[:,i_start:i_end] = self.cores[k]._read_matrix()
            return matrix

    def _save_matrix(self):
        '''
        Save the internal matrix held by this core
        Unlike _read_matrix, all data necessary to restore the matrix is provided.
        No quantization or other errors are applied.
        '''
        if self.Ncores == 1:
            return self.core._save_matrix()
        else:
            raise NotImplementedError("_save_matrix not yet supported for split MVM")


    def _restore_matrix(self, matrix):
        '''
        Restore an internal matrix held by this core (debug method)
        You should only use a matrix obtained from _save_matrix, as _read_matrix may remove needed values (e.g.: from an offset core).
        No quantization or other errors are applied.
        '''
        if self.Ncores == 1:
            self.core._restore_matrix(matrix)
        else:
            raise NotImplementedError("_restore_matrix not yet supported for split MVM")



    def reshape_input(self,M_input):
        '''
        Reshape a vector input to a matrix input with the dimensions specified for this conv layer. The vector must be the
        appropriate length
        '''

        Nix = self.params.convolution_parameters.Nix
        Niy = self.params.convolution_parameters.Niy
        Nic = self.params.convolution_parameters.Nic

        if M_input.shape == (Nic,Nix,Niy):
            return M_input

        # If input is a vector assume there is only one channel
        if len(M_input.shape) == 1 and Nic != 1:
            raise ValueError("Multiple input channels are needed for conv layer")

        # Try to avoid reshape operation in the channel dimension
        if len(M_input.shape) > 1 and M_input.shape[-1] != Nic:
            raise ValueError("Input does not have correct dimensions to be used in conv layer")

        try:
            return M_input.reshape(Nic,Nix,Niy)
        except:
            raise ValueError("Input does not have correct dimensions to be used in conv layer")
            

    def apply_convolution(self,M_input,profiling=False):
        '''
        Applies a convolution operation to an input matrix M_input. Uses the sliding window method. Each MVM returns returns the outputs ...
        for all output channels for a single window. The results are then re-constructed into an output matrix that follows the same ...
        format as the input matrix.

        M_input: a 3D matrix of size (Nx, Ny, Nic). The third dimension must match the Nic in the conv core's parameters
        M_output: a 3D matrix of size (Nx_out, Ny_out, Noc). Nx_out and Ny_out is a function of the conv core's filter size, stride, and padding
        '''

        Nic, Nx, Ny = M_input.shape

        # Attempt to reshape input if incorrect size
        if M_input.shape != (self.params.convolution_parameters.Nic,self.params.convolution_parameters.Nix,self.params.convolution_parameters.Niy):
            M_input = self.reshape_input(M_input)

        if Nic != self.params.convolution_parameters.Nic or Nx != self.params.convolution_parameters.Nix or Ny != self.params.convolution_parameters.Niy:
            raise ValueError("The number of channels in the input matrix does not match the number of input channels in the convolutional layer.")

        Kx = self.params.convolution_parameters.Kx
        Ky = self.params.convolution_parameters.Ky
        Noc = self.params.convolution_parameters.Noc
        Nrows = self.params.convolution_parameters.Nrows
        stride = self.params.convolution_parameters.stride
        px_0 = self.params.convolution_parameters.px_0
        px_1 = self.params.convolution_parameters.px_1
        py_0 = self.params.convolution_parameters.py_0
        py_1 = self.params.convolution_parameters.py_1
        x_par = self.params.numeric_params.x_par
        y_par = self.params.numeric_params.y_par
        weight_reorder = self.params.numeric_params.weight_reorder
        NrowsX = Kx*Ky*Nic # number of rows per sliding window MVM

        # Apply zero padding
        M_input = ncp.pad(M_input,((0,0),(px_0,px_1),(py_0,py_1)),'constant')

        # Number of sliding windows
        Nx_out, Ny_out = ((M_input.shape[1] - Kx)//stride + 1, (M_input.shape[2] - Ky)//stride + 1)

        # Allocate memory for the output
        M_out = ncp.empty((Noc,Nx_out,Ny_out),dtype=M_input.dtype)
        if profiling:
            M_out_all = [ncp.zeros((Noc,Nx_out,Ny_out)) for m in range(self.Ncores)]

        # Scale, clip and quantize the inputs all at once
        # I have found that this actually leads to a significant speedup (~20%) with the ADC enabled - PX
        mvm_scale = self.params.wrapper_params.col_input.range/self.params.algorithm_params.col_input.range
        M_input *= mvm_scale
        M_input = self.params.xbar_params.col_input.clip_and_quantize(M_input)
        if self.params.convolution_parameters.bias:
            mvm_scale = self.params.xbar_params.col_input.clip_and_quantize(ncp.array([mvm_scale]))[0]

        for i in range(0,Nx_out,x_par):
            x_block = (x_par if (Nx_out - i) >= x_par else Nx_out - i)
            for j in range(0,Ny_out,y_par):
                y_block = (y_par if (Ny_out - j) >= y_par else Ny_out - j)
                x_start = i*stride
                y_start0 = j*stride
                if Kx == 1 and Ky == 1:
                    Min_large = ncp.ones(int(Nrows*x_par*y_par),dtype=M_input.dtype)
                    Min_large *= mvm_scale
                    v_start, v_end = 0, NrowsX
                    for xp in range(x_par):
                        y_start = y_start0
                        for yp in range(y_par):
                            if xp <= (x_block-1) and yp <= (y_block-1):
                                Min_ij = M_input[:,x_start,y_start]
                                Min_large[v_start:v_end] = Min_ij
                            else:
                                Min_large[v_start:v_end] = 0
                            y_start += stride
                            v_start += Nrows
                            v_end += Nrows
                        x_start += stride
                else:
                    if weight_reorder:
                        x_end = x_start + Kx + stride*(x_par - 1)
                        y_end = y_start0 + Ky + stride*(y_par - 1)
                        Min_large = M_input[:,x_start:x_end,y_start0:y_end].flatten()

                    else:
                        Min_ij = ncp.zeros((Nic*x_par*y_par,Kx,Ky),dtype=M_input.dtype)
                        x_end = x_start + Kx
                        v_start, v_end = 0, Nic

                        for xp in range(x_par):
                            y_start = y_start0
                            y_end = y_start + Ky
                            for yp in range(y_par):
                                if xp <= (x_block-1) and yp <= (y_block-1):
                                    Min_ij[v_start:v_end,:,:] = M_input[:,x_start:x_end,y_start:y_end]
                                y_start += stride
                                y_end += stride
                                v_start += Nic
                                v_end += Nic
                            x_start += stride
                            x_end += stride

                        if self.params.convolution_parameters.bias:
                            Min_ij2 = ncp.ones((x_par*y_par,Nrows),dtype=M_input.dtype)
                            Min_ij2 *= mvm_scale
                            Min_ij = Min_ij.reshape((x_par*y_par,NrowsX))
                            Min_ij2[:,:-1] = Min_ij
                            Min_large = Min_ij2.flatten()
                        else:
                            Min_large = Min_ij.flatten()

                self.set_mvm_inputs(Min_large)

                if not profiling:
                    M_out_p = self.run_xbar_mvm()
                    # The line below is pure diabolical sorcery
                    if x_block == x_par and y_block == y_par:
                        M_out[:,i:(i+x_par),j:(j+y_par)] = M_out_p.reshape((Noc,y_par,x_par),order='F').transpose((0,2,1))
                    else:
                        M_out[:,i:(i+x_block),j:(j+y_block)] = M_out_p.reshape((Noc,y_par,x_par),order='F').transpose((0,2,1))[:,:x_block,:y_block]

                else:
                    M_out_p, M_out_p_all = self.run_xbar_mvm(profiling=True)
                    M_out[:,i:(i+x_par),j:(j+y_par)] = M_out_p.reshape((Noc,y_par,x_par),order='F').transpose((0,2,1))
                    for k in range(self.Ncores):
                        Mpk = M_out_p_all[k].copy()
                        M_out_all[k][:,i:(i+x_par),j:(j+y_par)] = Mpk.reshape((Noc,y_par,x_par),order='F').transpose((0,2,1))

        if not profiling:
            return M_out, None
        else:
            return M_out, M_out_all