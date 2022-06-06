#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Simple example for how to directly use a neural core without going through the
inference or training front end

**Note**: As of 05/2022, the available functionality is asymmetric between the
MVM (used for inference and training) and the VMM (used for training only).
The MVM currently supports a greater range of mapping options that are not yet
implemented in the VMM, such as input and weight bit slicing. This asymmetry
will be addressed in a future release.

This script includes only the limited set of functionalities available to both
MVM and VMM. The MVM-exclusive functionalities are not included here but are
used heavily as part of CrossSim Inference (see inference_net.py). This
script will be updated in the future to include these features once they are
implemented for both MVM and VMM.

'''
import sys
import numpy as np

# Import cross_sim
sys.path.append("..")
from cross_sim import MakeCore
from cross_sim import Parameters


def main():

    # ********************** set all parameters  ************************************************
    params = Parameters()  # create a parameter object with all the default settings

    # Specify the crossbar type, should be a BALANCED or OFFSET.  That determines how negative weights are handled
    params.algorithm_params.crossbar_type = "OFFSET"

    # ************** clip constraints (weights, read, driving)  max should equal minus min
    params.algorithm_params.disable_clipping = False # disable all clipping?

    params.algorithm_params.weights.maximum = 10
    params.algorithm_params.weights.minimum = -10
    # min weight can be zero when using offset core, otherwise they should be equal and opposite

    params.algorithm_params.col_input.maximum = 1.0
    params.algorithm_params.col_input.minimum = -1.0

    params.algorithm_params.row_input.maximum = 1.0
    params.algorithm_params.row_input.minimum = -1.0

    params.algorithm_params.col_output.maximum = 20
    params.algorithm_params.col_output.minimum = -20

    params.algorithm_params.row_output.maximum = 20
    params.algorithm_params.row_output.minimum = -20

    params.algorithm_params.col_update.maximum = 5
    params.algorithm_params.col_update.minimum = -5

    params.algorithm_params.row_update.maximum = 5
    params.algorithm_params.row_update.minimum = -5


    # *********************** A/D and D/A noise parameters  (shown for row input, but can be used for any part above)
    # **** bits for quantization
    params.xbar_params.row_input.bits = 0
    params.xbar_params.row_input.sign_bit = True  # total bits  =  bits + sign_bit

    # **** Gaussian noise applied after quantization
    params.xbar_params.row_input.normal_error_post.sigma = 0.0  # sigma defined relative to full range
    # new value = old value + N(sigma*range)

    params.xbar_params.row_input.normal_error_post.keep_within_range = True
    # clips the resulting value after applying noise to the clipping constraints

    params.xbar_params.row_input.normal_error_post.proportional = False
    # use proportional noise model instead
    # False: new value = old value + N(sigma*range)
    # True: new value = old value * [ 1+N(sigma) ]

    # **** Gaussian noise applied before quantization
    params.xbar_params.row_input.normal_error_pre.keep_within_range = False
    params.xbar_params.row_input.normal_error_pre.proportional = False
    params.xbar_params.row_input.normal_error_pre.sigma = 0.0

    # **** Uniform noise applied after quantization
    params.xbar_params.row_input.uniform_error_post.range = 0.0
    # the range of the noise in fraction of whole range

    params.xbar_params.row_input.uniform_error_post.keep_within_range = False
    # clips the resulting value to the clipping constraints


    # ********************* Matrix read noise *************************************
    # gaussian noise, parameters defined same as above for normal_error_post
    # only applied during VMM and MVM
    params.numeric_params.read_noise.sigma = 0
    params.numeric_params.read_noise.proportional = False
    params.numeric_params.read_noise.keep_within_range = True


    # ********************* Update Model Selection ********************************
    # Numerically, there are two udpate models for the weights, an analytic model or a experimental lookup table model
    params.numeric_params.update_model = "ANALYTIC"  # ANALYTIC  means use the analytic models (in Agarwal et all, IJCNN 2016)
    # "DG_LOOKUP"  This model uses a experimentally derived lookup table


    # **********************  Lookup Table Model **********************************
    # only used if setting above is "DG_LOOKUP"
    # only applied during update_matrix ( not set_matrix)

    params.numeric_params.dG_lookup.file_increasing = "LISTA_current"
    #  this is the lookup table for increasing pulses. Can be one of several different device lookup tables are stored in the data folder, or the path to a user defined file
    params.numeric_params.dG_lookup.file_decreasing = "LISTA_current"
    #  this is the lookup table for increasing pulses. Can be one of several different device lookup tables are stored in the data folder, or the path to a user defined file
    params.numeric_params.dG_lookup.Gmin_relative = 0.25
    params.numeric_params.dG_lookup.Gmax_relative = 0.75
    # the lookup table defines a large conductance range, but using only part of it may give better results
    # Gmin_relative and Gmax_relative define the fraction of the lookup table (25% to 75% of the range) to target using
    # weights can go outside the targeted range, but the weight scaling will be based on the targeted range.

    # ********************* Matrix write noise *************************************
    # only applied if update model is "ANALYTIC"
    # gaussian noise, with sigma defined as below
    # only applied during update_matrix ( not set_matrix)

    params.numeric_params.write_noise.sigma = 0
    params.numeric_params.write_noise.write_noise_model = "G_INDEPENDENT"
    # possible models:  (sigma_wn =  parameter set above)
    # "G_INDEPENDENT" # write noise independent of the conductance
    #       sigma = sqrt( delta_G*range)           *sigma_wn
    # "G_PROPORTIONAL" = 1 # write noise proportional to the current conductance
    #       sigma = sqrt( delta_G*range) * G/range *sigma_wn
    # "G_INVERSE" = 2 # write noise inversely proportional to the current conductance
    #       sigma = sqrt( delta_G*range) * range/G *sigma_wn


    # **************nonlinear update model  *************************
    # only applied if update model is "ANALYTIC"

    params.numeric_params.nonlinearity.alpha = 0
    # zero means no nonlinearity, larger number is worse nonlinearity
    params.numeric_params.nonlinearity.symmetric = False
    # symmetric or asymmetric nonlinearity


    # ********************  use the crossbar *******************************

    # set the array and vector

    vector = np.array([0, -0.5, 0, 1, 0, 0])

    weights = np.array([[1, 0, 0.4, 0.6, 0.8, 1],
                        [1, 0.2, 0.4, 0.6, 0.8, 1],
                        [1, 0.2, 0.4, 0.6, 0.8, 1],
                        [1, 0.2, 0.4, 0.6, 0.8, 1],
                        [1, 0.2, 0.4, 0.6, 0.8, 1],
                        [-1, -0.2, -0.4, -0.6, -0.8, -1]]) * 10


    # make the neural core and pass all the parameter settings
    neural_core = MakeCore(params=params)

    # lets do a vector matrix multiply
    neural_core.set_matrix(weights)

    print('Vector-Matrix Multiplication:')
    print(vector)
    print('.')
    print(weights)
    print('=')

    # actually run the vector matrix multiply
    print(neural_core.run_xbar_vmm(vector))
    print('exact result = ')
    print(np.dot(vector, weights))
    print()

    # now do the transpose: a matrix vector multiply

    # transpose the weights as well to get the same result
    weights2 = np.transpose(weights)
    neural_core.set_matrix(weights2)

    print('Matrix-Vector Multiplication:')
    print(weights2)
    print('.')
    print(vector)
    print('=')
    print(neural_core.run_xbar_mvm(vector))
    print('exact result = ')
    print(np.dot(weights2, vector))
    print()

    # do an outer product update

    neural_core.set_matrix(weights)
    vector1 = [1, 1, 1, 1, 0.2, 1]
    vector2 = [1, 0.5, 1, -1, 0, 0]

    print('Numeric (Outer-Product Update)')
    print(weights)
    print('+')
    print(str(vector1) + " outer " + str(vector2) + " which is")
    print(np.outer(vector1, vector2))
    print('=')
    neural_core.update_matrix(vector1, vector2)
    print(neural_core._read_matrix())

    print('exact result = ')
    print(weights + np.outer(vector1, vector2))
    print('The neural core weights are limited to +/-10 and are therefore clipped')
    print()


    # useful functions are:
    # set_mvm_inputs
    # run_xbar_vmm
    # run_xbar_mvm
    # update_matrix
    # serial_update
    # serial_read
    # _read_matrix (ideally only for debugging)
    # save_weights
    # load_weights


    print('done')

# 4. Call main()
if __name__ == '__main__':
    main()
