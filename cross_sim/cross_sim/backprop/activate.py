#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Activation models, implemented numerically

import numpy as np

STYLES = ("SIGMOID", "SIGMOIDSLOPE", "RECTLINEAR","SOFTMAX", "SHIFTED_SIGMOID","WHETSTONE","QUANTIZED_RELU","SIGN","NONE")
SIGMOID, SIGMOIDSLOPE, RECTLINEAR, SOFTMAX, SHIFTED_SIGMOID,WHETSTONE,QUANTIZED_RELU,SIGN,NONE = (0,1,2,3,4,5,6,7,8)

BIGVALUE = 1.0e20
extreme_counter = 0


# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)


# SciPy sigmoid function

# try:
#     from scipy.special import expit as sigmoid
# except ImportError:
#     print("WARNING: could not load SciPy sigmoid() function")


def sigmoid(x, **kwargs):

    y = -x
    y = ncp.exp(y, out=y)
    y += 1
    y = ncp.reciprocal(y, out=y)
    return y
    # return np.vectorize(lambda y: 1.0 / (1.0 + exp(-y)))(x)

#
# # sigmoid function
#
# def vanilla_sigmoid(x):
#     global extreme_counter
#     try:
#         return 1.0 / (1.0 + exp(-x))
#     except OverflowError:
#         extreme_counter += 1
#         if x > 0.0:
#             return 1.0
#         else:
#             return 0.0
#
#
# # derivative of sigmoid function
#
# def vanilla_sigmoidprime(x):
#     y = sigmoid(x)
#     return y * (1.0 - y)
#
#
# # user-slope sigmoid function
#
# def vanilla_slope_sigmoid(x, slope):
#     global extreme_counter
#     try:
#         return 1.0 / (1.0 + exp(-slope * x))
#     except OverflowError:
#         extreme_counter += 1
#         if x > 0.0:
#             return 1.0
#         else:
#             return 0.0
#
#
# # derivative of user-slope sigmoid function
#
# def vanilla_slope_sigmoidprime(x, slope):
#     y = vanilla_slope_sigmoid(x, slope)
#     return slope * y * (1.0 - y)


# # step function
#
# def step(x):
#     ret = np.sign(x)
#     ret[ret == 0.0] = 1.0
#     return ret


# numeric Activation models

class Activate:
    """
    activation models
    required args:
      style = SIGMOID for sigmoidal
              SIGMOIDSLOPE for sigmoidal with specified slope
              RECTLINEAR for rectifiead linear with leaky term
              could add STEP, but need derivative
    """

    def __init__(self, **kwargs):
        if "style" not in kwargs: error("Style setting required")
        self.style = kwargs["style"]

        # optional args

        self.sigslope = 1.0
        self.leakyslope = 0.0
        self.shift=0.0
        self.relu_bound = BIGVALUE
        self.sharpness=0.0
        self.useGPU=False
        self.nbits = 8
        if "sigslope" in kwargs: self.sigslope = kwargs["sigslope"]
        if "leakyslope" in kwargs: self.leakyslope = kwargs["leakyslope"]
        if "shift" in kwargs: self.shift = kwargs["shift"]
        if "relu_bound" in kwargs: self.relu_bound = kwargs["relu_bound"]
        if "sharpness" in kwargs: self.sharpness = kwargs["sharpness"]
        if "nbits" in kwargs: self.nbits = kwargs["nbits"]
        if "useGPU" in kwargs: self.useGPU = kwargs["useGPU"]

        global ncp
        if self.useGPU:
            import cupy as cp
            ncp = cp
        else:
            ncp = np

        # error checks

        if self.style not in STYLES: error("Unknown style")
        self.style = STYLES.index(self.style)

        if self.sigslope <= 0.0: error("Sigslope cannot be <= 0.0")
        if self.leakyslope < 0.0: error("Leakyslope cannot be < 0.0")

    # apply activation function, used in forward direction
    # does NOT modify x, returns new vector

    def apply(self, x):
        """

        :param x: input vector of all sums fed into activation function
        :return:
        """
        style = self.style
        if style == SIGMOID:
            return sigmoid(x)
        elif style == SIGMOIDSLOPE:
            return sigmoid(self.sigslope * x)
        elif style == SHIFTED_SIGMOID:
            return sigmoid(x-self.shift)

        elif style == RECTLINEAR:
            posvalues = x.clip(0.0, self.relu_bound)
            negvalues = x.clip(-self.relu_bound, 0.0)
            negvalues *= self.leakyslope
            posvalues += negvalues
            return posvalues
        elif style == SOFTMAX:
            result = ncp.exp(x-x.max())
            result = result/ncp.sum(result)
            return result
        elif style == WHETSTONE:
            if self.sharpness == 1.0:
                y = ncp.zeros(x.shape)
                y[x < 0.5] = 0
                y[x >= 0.5 ] = 1
                return y
            else:
                width = 1.0 - self.sharpness
                epsilon = 0.001
                return ncp.clip((1.0 / (width + epsilon)) * (x - 0.5) + 0.5, 0.0, 1.0)
        elif style == QUANTIZED_RELU:
            R = ncp.rint(ncp.clip((x+1)/2,0,1) * pow(2,self.nbits))
            y = ncp.clip( 2.*(R/pow(2,self.nbits))-1., 0, 1 - 1.0/pow(2,self.nbits-1))
            return y
        elif style == SIGN:
            return ncp.sign(x)
        elif style == NONE:
            return x
        else:
            raise ValueError("Undefined activation style ", style)

    # derivative of activation function, used in backward direction
    # does NOT modify x, returns new vector

    def derivative(self, x):
        style = self.style
        if style == SIGMOID:
            y = sigmoid(x)
            y *= (1.0 - y)
            return y
            # return np.vectorize(lambda z: z * (1.0 - z))(y)
        elif style == SIGMOIDSLOPE:
            y = sigmoid(self.sigslope * x)
            y *= self.sigslope * (1.0 - y)
            return y
        if style == SHIFTED_SIGMOID:
            y = sigmoid(x-self.shift)
            y *= (1.0 - y)
            return y
            # return np.vectorize(lambda z: z * (1.0 - z))(y)
        elif style == RECTLINEAR:  # is this efficient Numpy code?
            y = x.copy()
            y[y >= 0.0] = 1.0
            y[y < 0.0] = self.leakyslope
            return y
        elif style == SOFTMAX:
            return x*(1-x)
        else:
            raise ValueError("Undefined activation style ", style)
