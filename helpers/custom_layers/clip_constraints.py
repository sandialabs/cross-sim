from __future__ import absolute_import
from __future__ import print_function

import pkg_resources
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint

##  Keras constraints needed to call load_model on some pre-trained models
##  These constraints were used during training only and are not relevant for the inference simulation

class ClipPrecision(Constraint):
    """ Clips the precision of values by rounding.

        Note that ``bits`` should include the sign bit.
    """

    def __init__(self, 
                 clip_val = 1.0, 
                 bits     = 9):
        self.clip_val = clip_val
        self.bits     = bits

    def __call__(self, p):
        clipped = K.clip(p, -self.clip_val, self.clip_val)   
        return K.round(clipped * 2**self.bits) / 2**self.bits

    def get_config(self):
        config =  {'clip_val': self.clip_val,
                   'bits':     self.bits}
        config.update(super(ClipPrecision, self).get_config())
        return config


class Clip(Constraint):
    """ Clips the precision of values by rounding.

        Note that ``bits`` should include the sign bit.
    """

    def __init__(self, clip_val = 1.0):
        self.clip_val = clip_val

    def __call__(self, p):
        clipped = K.clip(p, -self.clip_val, self.clip_val)   
        return clipped

    def get_config(self):
        config =  {'clip_val': self.clip_val}
        config.update(super(Clip, self).get_config())
        return config


customObjects = {'ClipPrecision':ClipPrecision, 'Clip':Clip}
