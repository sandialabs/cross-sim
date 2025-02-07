#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from ...backend import ComputeBackend

xp = ComputeBackend()

STYLES = (
    "SIGMOID",
    "SIGMOIDSLOPE",
    "RECTLINEAR",
    "SOFTMAX",
    "SHIFTED_SIGMOID",
    "WHETSTONE",
    "QUANTIZED_RELU",
    "SIGN",
    "TANH",
    "NONE",
)
(
    SIGMOID,
    SIGMOIDSLOPE,
    RECTLINEAR,
    SOFTMAX,
    SHIFTED_SIGMOID,
    WHETSTONE,
    QUANTIZED_RELU,
    SIGN,
    TANH,
    NONE,
) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

BIGVALUE = 1.0e20
extreme_counter = 0


# Error output
def error(str, *args, **kwargs):
    raise ValueError(str, *args, **kwargs)


# Sigmoid function
def sigmoid(x, **kwargs):
    y = -x
    y = xp.exp(y, out=y)
    y += 1
    y = xp.reciprocal(y, out=y)
    return y


class Activate:
    """Activation models, implemented numerically."""

    def __init__(self, **kwargs):
        if "style" not in kwargs:
            error("Style setting required")
        # Required
        self.style = kwargs["style"]

        # Optional arguments
        self.sigslope = 1.0
        self.leakyslope = 0.0
        self.shift = 0.0
        self.relu_bound = BIGVALUE
        self.sharpness = 0.0
        self.nbits = 8
        if "sigslope" in kwargs:
            self.sigslope = kwargs["sigslope"]
        if "leakyslope" in kwargs:
            self.leakyslope = kwargs["leakyslope"]
        if "shift" in kwargs:
            self.shift = kwargs["shift"]
        if "relu_bound" in kwargs:
            self.relu_bound = kwargs["relu_bound"]
        if "sharpness" in kwargs:
            self.sharpness = kwargs["sharpness"]
        if "nbits" in kwargs:
            self.nbits = kwargs["nbits"]

        if self.style not in STYLES:
            error("Unknown style")
        self.style = STYLES.index(self.style)

        if self.sigslope <= 0.0:
            error("Sigslope cannot be <= 0.0")
        if self.leakyslope < 0.0:
            error("Leakyslope cannot be < 0.0")

    # apply activation function, used in forward direction
    # does NOT modify x, returns new vector

    def apply(self, x):
        """:param x: input vector of all sums fed into activation function
        :return:
        """
        style = self.style
        if style == SIGMOID:
            return sigmoid(x)

        elif style == SIGMOIDSLOPE:
            return sigmoid(self.sigslope * x)

        elif style == SHIFTED_SIGMOID:
            return sigmoid(x - self.shift)

        elif style == RECTLINEAR:
            posvalues = x.clip(0.0, self.relu_bound)
            negvalues = x.clip(-self.relu_bound, 0.0)
            negvalues *= self.leakyslope
            posvalues += negvalues
            return posvalues

        elif style == SOFTMAX:
            result = xp.exp(x - x.max())
            result = result / xp.sum(result)
            return result

        elif style == WHETSTONE:
            if self.sharpness == 1.0:
                y = xp.zeros(x.shape)
                y[x < 0.5] = 0
                y[x >= 0.5] = 1
                return y
            else:
                width = 1.0 - self.sharpness
                epsilon = 0.001
                return xp.clip((1.0 / (width + epsilon)) * (x - 0.5) + 0.5, 0.0, 1.0)

        elif style == QUANTIZED_RELU:
            R = xp.rint(xp.clip((x + 1) / 2, 0, 1) * pow(2, self.nbits))
            y = xp.clip(
                2.0 * (R / pow(2, self.nbits)) - 1.0,
                0,
                1 - 1.0 / pow(2, self.nbits - 1),
            )
            return y

        elif style == SIGN:
            return xp.sign(x)

        elif style == TANH:
            return xp.tanh(x)

        elif style == NONE:
            return x

        else:
            raise ValueError("Undefined activation style ", style)
