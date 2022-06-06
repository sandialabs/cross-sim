"""
This script builds a ResNet50-v1.5 Keras model with weights loaded from a separate file
Adapted from keras.applications.Resnet50
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

Author: T. Patrick Xiao
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras import backend as backend
import tensorflow.keras.models as models
import numpy as np
from scipy.interpolate import interp1d

class Quantize(keras.layers.Layer):

    def __init__(self,shift_bits,output_bits,signed=False,name=None):
        super(Quantize, self).__init__(name=name)
        
        self.signed = signed
        self.shift_bits = shift_bits
        self.output_bits = output_bits

        full = tf.pow(2,shift_bits)
        self.full = tf.cast(full,tf.float32)
        self.half = tf.divide(self.full,2.0)
        
        if signed:
            ymin = -tf.pow(2,output_bits)
            ymax = tf.pow(2,output_bits)-1
        else:
            ymin = tf.constant(0)
            ymax = tf.pow(2,output_bits)-1
        
        self.ymin = tf.cast(ymin,tf.float32)
        self.ymax = tf.cast(ymax,tf.float32)

    def build(self,input_shape):
        self.requant = self.add_weight(
            shape=(input_shape[-1],), initializer="zeros",trainable=False)

    def get_config(self):
        config = super(Quantize, self).get_config()
        config.update({"shift_bits": self.shift_bits, "output_bits": self.output_bits, "signed": self.signed})
        return config

    def call(self, inputs):

        if not self.signed:
            y = tf.add(tf.multiply(inputs,self.requant), self.half)
            y = tf.divide(y, self.full)
            y = tf.math.floor(y)

        else:
            pos = tf.cast(tf.math.greater_equal(inputs,0.0),tf.float32)
            neg = tf.cast(tf.math.less(inputs,0.0),tf.float32)
            y_pos = tf.add(tf.multiply(inputs,self.requant), self.half)
            y_pos = tf.divide(y_pos, self.full)
            y_pos = tf.math.floor(y_pos)
            y_neg = tf.subtract(tf.multiply(inputs,self.requant), self.half)
            y_neg = tf.divide(y_neg, self.full)
            y_neg = tf.math.ceil(y_neg)
            y = y_pos*pos + y_neg*neg

        # y = tf.multiply(inputs,self.requant)
        # y = tf.divide(y, self.full)
        # y = tf.math.round(y)

        # Truncate
        y = tf.clip_by_value(y, clip_value_min=self.ymin, clip_value_max=self.ymax)

        return y


class Scale(keras.layers.Layer):

    def __init__(self,classes=1000,name=None):
        super(Scale, self).__init__(name=name)
        self.units = classes
        self.W = self.add_weight(
            shape=(classes,), initializer="ones",trainable=False)

    def get_config(self):
        config = super(Scale, self).get_config()
        config.update({"units": self.units})
        return config

    def call(self, inputs):

        # return tf.multiply(inputs,self.W)
        return tf.divide(inputs,self.W)
        # return inputs


class Round(keras.layers.Layer):

    def __init__(self,name=None):
        super(Round, self).__init__(name=name)

    def call(self, inputs):

        # y = tf.math.floor(inputs + 0.5)
        y = tf.math.floor(inputs)

        return y


class Inspect(keras.layers.Layer):

    def __init__(self,name=None):
        super(Inspect, self).__init__(name=name)

    def call(self, inputs):

        y = backend.print_tensor(inputs)

        return y



def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    pad_name_base = 'pad' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = Quantize(name=conv_name_base + '2a_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name=pad_name_base + '2b')(x)
    x = layers.Conv2D(filters2, kernel_size,
                      padding='valid',
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2b')(x)

    x = layers.Activation('relu')(x)
    x = Quantize(name=conv_name_base + '2b_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2c')(x)

    shortcut = Quantize(name=conv_name_base + '1_quantize',shift_bits=1,output_bits=31,signed=False)(input_tensor)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    if stage == 5 and block == 'c':
        x = Quantize(name=conv_name_base + '2c_quantize',shift_bits=16,output_bits=7,signed=False)(x)
    else:
        x = Quantize(name=conv_name_base + '2c_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    pad_name_base = 'pad' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), 
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = Quantize(name=conv_name_base + '2a_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name=pad_name_base + '2b')(x)
    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='valid',
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2b')(x)


    x = layers.Activation('relu')(x)
    x = Quantize(name=conv_name_base + '2b_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',use_bias=True,
                      name=conv_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',use_bias=True,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = Quantize(name=conv_name_base + '1_quantize1',shift_bits=16,output_bits=7,signed=True)(shortcut)

    if stage == 2 or stage == 3:
        shortcut = Quantize(name=conv_name_base + '1_quantize2',shift_bits=0,output_bits=31,signed=True)(shortcut)
    elif stage == 4:
        shortcut = Quantize(name=conv_name_base + '1_quantize2',shift_bits=4,output_bits=31,signed=True)(shortcut)
    elif stage == 5:
        shortcut = Quantize(name=conv_name_base + '1_quantize2',shift_bits=10,output_bits=31,signed=True)(shortcut)


    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    x = Quantize(name=conv_name_base + '2c_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    return x


def ResNet50v15_int4(weight_dict,quant_dict,classes=1000,**kwargs):
    """Instantiates the ResNet50v1.5 architecture.
    # Arguments
        weight_dict: dictionary of weight matrices accessed with the layer
            name. Weight matrices are stored in the format returned by get_weights()
        classes: 1000 for Nvidia's ResNet50-v1.5 INT4 model
    # Returns
        A Keras model instance with loaded weights.
    """

    # Determine proper input shape
    input_shape = (224,224,3)

    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1',use_bias=True)(x)
    x = layers.Activation('relu')(x)
    x = Quantize(name='conv1_quantize',shift_bits=16,output_bits=4,signed=False)(x)

    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

  
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = Round(name='round')(x)
    x = layers.Dense(classes, name='fc1000',use_bias=True)(x)
    x = Scale(classes, name='scale')(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x, name='resnet50')

    for i in range(len(model.layers)):
        layer_name = model.layers[i].get_config()['name']

        if layer_name in weight_dict:

            # Check for batchnorm
            if layer_name == 'conv1' or 'res' in layer_name or 'fc' in layer_name or layer_name == 'scale':

                weights = weight_dict[layer_name]
                Wm = weights[0]

                if len(weights) > 1:
                    Wbias = weights[1]
                    weights = [Wm, Wbias]

                else:
                    weights = [Wm]

            model.layers[i].set_weights(weights)

        if layer_name in quant_dict:
            weights = [quant_dict[layer_name][0]]
            model.layers[i].set_weights(weights)

    return model