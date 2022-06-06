#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import ZeroResetPCEnum
from warnings import warn
# from .parameter_defaults import NeuronStyleEnum


class ConvolutionParameters(ParametersBase):

    if False:
        is_conv_core=bool
        sameConv=bool # pad input so conv does not change size of each channel
        stride=int
        px_0=int
        px_1=int
        py_0=int
        py_1=int
        Kx=int # conv filter size x
        Ky=int # conv filter size y
        Noc=int # output channels
        Nix=int
        Niy=int
        Nic=int # input channels
        bias=bool # whether to have a bias row
        Nrows=int # number of rows in array
        subarray_id=int # if matrix is split along row dimension, which core it is

    # style = Parameter (name = "style", value = None, post_set= ParametersBase.generate_enum_post_set("style", NeuronStyleEnum) )

    def post_set(self):
        self.override_readonly = True

        # Calculate number of rows
        if self.bias:
            self.Nrows = self.Kx*self.Ky*self.Nic + 1
        else:
            self.Nrows = self.Kx*self.Ky*self.Nic

        # Make sure input dimensions are compatible with filter
        # if self.Nix < self.Kx or self.Niy < self.Ky:
        #     raise ValueError('Conv input size incompatible with filter size')

        # Calculate output size
        if self.sameConv:
            (self.Nox,self.Noy) = (self.Nix//self.stride,self.Niy//self.stride)
        else:
            self.Nox = 1 + (self.Nix-self.Kx+self.px_0+self.px_1)//self.stride
            self.Noy = 1 + (self.Niy-self.Ky+self.py_0+self.py_1)//self.stride

        # Number of sliding windows
        self.Nwindows = self.Nox * self.Noy

        # If sameConv, calculate padding
        if self.sameConv:
            if (self.Kx % 2 != 0) and (self.Ky % 2 != 0):
                if (self.Nix % self.stride == 0):
                    px = max(self.Kx - self.stride, 0)
                else:
                    px = max(self.Kx - (self.Nix % self.stride), 0)
                if (self.Niy % self.stride == 0):
                    py = max(self.Ky - self.stride, 0)
                else:
                    py = max(self.Ky - (self.Niy % self.stride), 0)
                self.px_0 = px // 2
                self.px_1 = px - self.px_0
                self.py_0 = py // 2
                self.py_1 = py - self.py_0
            else:
                # Even size filter
                px = (self.Nox - 1)*self.stride + self.Kx - self.Nix
                py = (self.Noy - 1)*self.stride + self.Ky - self.Niy
                self.px_0 = px // 2
                self.px_1 = px - self.px_0 # px_1 is one more than px_0 if px is odd
                self.py_0 = py // 2
                self.py_1 = py - self.py_0


        # Warn if conv stride causes some edge rows or columns to be discarded
        # if not self.sameConv and ( (self.Nix-self.Kx+2*self.px_0) % self.stride != 0 or (self.Niy-self.Ky+2*self.py_0) % self.stride != 0):
        #     warn("Some rows/cols of matrix will be cut off using the specified conv stride")

        # Warn if maxpool stride causes some edge rows or columns to be discarded
        # if (Nox - self.MPx) % self.stride_MP != 0 or (Noy - self.MPy) % self.stride_MP != 0:
        #     warn("Some row/cols of matrix will be cut off using the specified maxpool stride")

        self.override_readonly = False

    # set readonly vars
    # Leave Nrows writable to allow matrix splitting across multiple cores
    Nrows = Parameter(name="Nrows", readonly=False)
    Nox = Parameter(name="Nox", readonly=True)
    Noy = Parameter(name="Noy", readonly=True)

    def __init__(self, param_root):
        #load the defaults
        ParametersBase.__init__(self,param_root, **params.ConvolutionDefaults)
