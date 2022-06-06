##
##   The following code has been adapted from Sandia's open-source Whetstone package
##   to be compatible with tensorflow.keras (rather than keras)
##
##   Source: https://github.com/SNL-NERL/Whetstone
##

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer
import tensorflow as tf 
import numpy as np


def _key_check(key, size):
    if (key is None):
        if (size is not None):
            return key_generator(size[0], size[1])
        else:
            raise ValueError("You must specifiy a key or a size tuple.")
    else:
        return key

class Spiking(Layer):
    """Abstract base layer for all spiking activation Layers.
    
    This layer should not be instantiated, but rather inherited.
    
    # Arguments
        sharpness: Float, abstract 'sharpness' of the activation.
            Setting sharpness to 0.0 leaves the activation function unmodified.
            Setting sharpness to 1.0 sets the activation function to a threshold gate.
    """
    sharpen_start_limit = 0.0
    sharpen_end_limit = 1.0

    def __init__(self, sharpness=0.0, activity_regularizer=None, **kwargs):
        super(Spiking, self).__init__(**kwargs)
        self.supports_masking = True
        self.sharpness = K.variable(K.cast_to_floatx(sharpness))
        if type(activity_regularizer) in [PenalizeCentrism]:
            self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        super(Spiking, self).build(input_shape)

    def sharpen(self, amount=0.01):
        """Sharpens the activation function by the specified amount.

        # Arguments
            amount: Float, the amount to sharpen.
        """
        K.set_value(self.sharpness, min(max(K.get_value(self.sharpness) + amount, Spiking.sharpen_start_limit),
                                        Spiking.sharpen_end_limit))

    def get_config(self):
        """ Provides configuration info so model can be saved and loaded.

        # Returns
            A dictionary of the layer's configuration.
        """
        config = {'sharpness': K.get_value(self.sharpness)}
        base_config = super(Spiking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Spiking_BRelu(Spiking):
    """ A Bounded Rectified Linear Unit layer that can be sharpened to a threshold gate.

        The sharpness value of the layer is inverted to determine the width of the 
        linear-region (i.e. non-binary region), which determines the slope 
        of the line in the linear-region such that the line intersects y = 0 and y = 1
        at the current step-function borders. The line will always pass through the
        point (0.5, 0.5).
    """

    def __init__(self, **kwargs):
        super(Spiking_BRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Spiking_BRelu, self).build(input_shape)

    def call(self, inputs):
        step_function = K.cast(K.greater_equal(inputs, 0.5), K.floatx())
        width = 1.0 - self.sharpness  # width of 'non-binary' region.
        epsilon = 0.001
        pbrelu = K.clip((1.0 / (width + epsilon)) * (inputs - 0.5) + 0.5, 0.0, 1.0)
        return K.switch(K.equal(self.sharpness, 1.0), step_function, pbrelu)

    def get_config(self):
        base_config = super(Spiking_BRelu, self).get_config()
        return dict(list(base_config.items()))


class PenalizeCentrism(Regularizer):
    """Regularizer that penalizes activations in proportion to how close they are to 0.5.

    Important: Assumes activations are always between 0.0 and 1.0.

    # Arguments
        strength: float between 0.0 and 1.0 which is multplied by the penalty.
    """

    def __init__(self, strength=0.0):
        super(Regularizer, self).__init__()
        self.strength = K.variable(K.cast_to_floatx(strength))

    def __call__(self, x):
        regularization = K.mean(0.5 - K.abs(x - 0.5)) * (self.strength * 2.0)
        return regularization

    def get_config(self):
        config = {'strength': K.get_value(self.strength)}
        # config.update(super(PenalizeCentrism, self).get_config())
        return config


class Softmax_Decode(Layer):
    """ A layer which uses a key to decode a sparse representation into a softmax.

    Makes it easier to train spiking classifiers by allowing the use of  
    softmax and catagorical-crossentropy loss. Allows for encodings that are 
    n-hot where 'n' is the number of outputs assigned to each class. Allows
    encodings to overlap, where a given output neuron can contribute 
    to the probability of more than one class.

    # Arguments
        key: A numpy array (num_classes, input_dims) with an input_dim-sized
            {0,1}-vector representative for each class.
        size: A tuple (num_classes, input_dim).  If ``key`` is not specified, then
            size must be specified.  In which case, a key will automatically be generated.
    """

    def __init__(self, key=None, size=None, **kwargs):
        super(Softmax_Decode, self).__init__(**kwargs)
        self.key = _key_check(key, size)
        if type(self.key) is dict and 'value' in list(self.key.keys()):
            self.key = np.array(self.key['value'], dtype=np.float32)
        elif type(self.key) is list:
            self.key = np.array(self.key, dtype=np.float32)
        # self._rescaled_key = K.variable(np.transpose(2*self.key-1))
        self._rescaled_key = K.variable(2 * np.transpose(self.key) - 1)

    def build(self, input_shape):
        super(Softmax_Decode, self).build(input_shape)

    def call(self, inputs):
        # return K.softmax(K.dot(2*(1-inputs),self._rescaled_key))
        return K.softmax(K.dot(2 * inputs - 1, self._rescaled_key))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.key.shape[0])

    def get_config(self):
        base_config = super(Softmax_Decode, self).get_config()
        return dict(list(base_config.items()) + [('key', self.key)])