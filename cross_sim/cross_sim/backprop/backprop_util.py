#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from scipy.special import softmax

def init_GPU_util(useGPU):
    global ncp
    global as_strided
    if useGPU:
        from cupy.lib.stride_tricks import as_strided
        import cupy as cp
        ncp = cp
    else:
        from numpy.lib.stride_tricks import as_strided
        ncp = np


# Function for applying max and average pooling
def apply_pool(matrix,MPx,MPy,stride_MP,poolType,px_L,px_R,py_L,py_R,avgPool_round,useGPU):
    '''
    Perform a max pool operation on a matrix, taking the max over MPx x MPy block with a given stride
    Assume the stride is equal to the kernel size
    This function is not tied to the convolution core object
    '''
    # No maxpool case

    # Padding
    if px_L > 0 or px_R > 0 or py_L > 0 or py_R > 0:
        matrix = ncp.pad(matrix,((0,0),(px_L,px_R),(py_L,py_R)),'constant')

    if MPx == 1 and MPy == 1 and stride_MP == 1:
        return matrix

    Nc, Nx, Ny = matrix.shape

    # If matrix size is not divisible by MaxPool stride, cut off bottom and/or right edges of matrix
    # This is the TensorFlow/Keras convention
    if Nx % stride_MP != 0 or Ny % stride_MP != 0:
        x_extra = Nx % stride_MP
        y_extra = Ny % stride_MP
        matrix = matrix[:,:(Nx-x_extra),:(Ny-y_extra)]

    if MPx == stride_MP and MPy == stride_MP:
        # This is slightly faster for non-overlapping pooling (the common case)
        Bx = Nx // MPx
        By = Ny // MPy
        if poolType == "max":
            return matrix.reshape(Nc,Bx,MPx,By,MPy).max(axis=(2,4))
        elif poolType == "avg":
            if not avgPool_round:
                return matrix.reshape(Nc,Bx,MPx,By,MPy).mean(axis=(2,4))
            else:
                return ncp.floor(matrix.reshape(Nc,Bx,MPx,By,MPy).mean(axis=(2,4)))

    else:        
        # Re-written for channel first order
        Mout_shape = (Nc,(Nx - MPx)//stride_MP + 1, (Ny - MPy)//stride_MP + 1)
        kernel_size = (1, MPx, MPy)
        M0 = as_strided(matrix, shape = Mout_shape+kernel_size, 
            strides = (matrix.strides[0], stride_MP*matrix.strides[1], stride_MP*matrix.strides[2]) + matrix.strides)
        M0 = M0.reshape(-1, *kernel_size)
        if poolType == "max":
            return M0.max(axis=(2,3)).reshape(Mout_shape)
        elif poolType == "avg":
            if not avgPool_round:
                return M0.mean(axis=(2,3)).reshape(Mout_shape)
            else:
                return ncp.floor(M0.mean(axis=(2,3)).reshape(Mout_shape))


def flatten_layer(matrix, useGPU):
    '''
    Flatten a 3D matrix (Nx,Ny,Nchannels) to a 1D vector, in a way that is identical with a Flatten layer in Keras
    '''
    matrix = ncp.transpose(matrix,(1,2,0))
    if useGPU:
        return matrix.flatten()
    else:
        return matrix.flatten("C")

# Applies a space2depth operation, assuming NCHW data format
# See: https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth
def space_to_depth(x, block_size, useGPU):

    x = x.transpose((1,2,0))
    height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(reduced_height, block_size,reduced_width, block_size, depth)
    z = np.swapaxes(y,1,2).reshape(reduced_height, reduced_width, -1)
    z = z.transpose((2,0,1))
    return z

# Nvidia custom quantization/de-quantization layer
# Applies fake quantization
# See: https://github.com/mlperf/inference_results_v0.5/tree/master/open/NVIDIA
def apply_quantization(x, W, shift_bits, output_bits, signed, useGPU):
    
    ymax = pow(2,output_bits)-1
    if not signed:
        ymin = 0
        y = x * W + pow(2,shift_bits)/2
        y /= pow(2,shift_bits)
        y = ncp.floor(y)

    else:
        ymin = -pow(2,output_bits)
        y_pos = x * W + pow(2,shift_bits)/2
        y_pos /= pow(2,shift_bits)
        y_pos = ncp.floor(y_pos)
        y_neg = x * W - pow(2,shift_bits)/2
        y_neg /= pow(2,shift_bits)
        y_neg = ncp.ceil(y_neg)
        y = y_pos * (x >= 0) + y_neg * (x < 0)

    y.clip(ymin, ymax, out=y)

    return y


# Output function for Whetstone models

def decode_from_key(key, input_vec):
    """Decodes a vector using the specified key.

    # Arguments
        key: Key used for decoding (ndarray)
        input_vec: Vector of size key.shape[1] to be decoded.

    # Returns
        Decoded one-hot vector.
    """
    # return [1*(np.argmax(np.matmul(2*key-1,2*input_vec-1))==i) for i in range(0, key.shape[0])]
    return softmax(np.dot(2*input_vec - 1, 2*key - 1))