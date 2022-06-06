#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""
All clipping constraints must have equal max and min magnitudes.  Clipping may fail if magnitudes are not equal:
"""
#TODO: Force max and min to be the same magnitude or else fix scaling to allow for different max and min for clipping constraints

import enum
import math

# ***************************************** define enumerated options **************************************************
class DriverStyleEnum(enum.IntEnum):
    '''
    Specifies the type of write driver to instantiate (pulse timing based or voltage level based)
    '''
    TIMED = 0
    LEVEL = 1

class NeuronStyleEnum(enum.Enum):
    RESISTOR = 'RESISTOR'
    VCVS_R = 'VCVS-R'
    VCVS_C = 'VCVS-C'


class DeviceModelEnum(enum.Enum):
    YAKOPCIC = 1  #yakopcic model implemented in xyce
    PEM = 2  # Piecewise empirical model

class WriteNoiseModelEnum(enum.IntEnum):
    G_INDEPENDENT = 0  # write noise independent of the conductance
        # sigma = sqrt( delta_G*range)           *sigma_wn
    G_PROPORTIONAL = 1 # write noise proportional to the current conductance
        # sigma = sqrt( delta_G*range) * G/range *sigma_wn
    G_INVERSE = 2 # write noise inversely proportional to the current conductance
        # sigma = sqrt( delta_G*range) * range/G *sigma_wn


class MemoryReadModelEnum(enum.IntEnum):
    ALL_GND = 0  # V/2 or all unselected grounded
    # DEFINED = 1 # defined model where all voltages are user specified


class ParasiticCompensationEnum(enum.IntEnum): #what method of parasitic compensation to use
    NONE = 0
    SERIAL = 1
    PARALLEL = 2
    PARTIALLY_PARALLEL = 3

class XyceTimeSteppingEnum(enum.IntEnum): #how many simulation time steps to use
    FULL = 0  # use the full default simulation time steps
    REDUCED = 1 # a reduced set determined by the breakpoints
    NOBREAKOPOINTS = 2 # ultra reduced simulation time points, does not capture power supply edges

class UpdateModelEnum(enum.IntEnum): # what type of update model to use
    ANALYTIC = 0  # use the analytic nonlinearity and write noise models
    DG_LOOKUP = 1 # use the the dG lookup table
    PCM = 2 # use PCM analytic model from IBM

class ZeroResetPCEnum(enum.IntEnum):
    EXACT=0
    CALIBRATED=1
    BLIND=2

class CrossbarTypeEnum(enum.IntEnum):
    MEMORY=0
    OFFSET=1
    BALANCED=2
    BITSLICED=3

class SimTypeEnum(enum.IntEnum):
    NUMERIC=0
    XYCE=1
    HARDWARE = 2


# ***************************************** define parameters **********************************************************

HardwareParametersDefaults = {
    'binary_updates': True, # are the updates limited to only binary values such that multi bit updates must be done as a series of pulses
    'relative_update_size' : 0.001, # what is the size of an update relative to a default update
    'set_matrix_error' : .05
}


PeriodicCarryDefaults = {
    'use_periodic_carry':False, # whether to use periodic carry
    'cores_per_weight':3,  #the number of cores used to represent the weights
    'number_base' : 5,  # the number system used between ReRAMs representing a weight (i.e base 5 or base 10 number system)
    'carry_threshold':0.5,  # at what fraction of the weight range should a carry be performed (wieghts less than this fraction are not carried)
    'normalized_output_scale':10,  # scale for which lower order bit outputs should be read.  The final scale is input range* weight range*normalized scale
    'read_low_order_bits': True,  # include lower order bits when reading.  Could get errors due to bits saturating
    'carry_frequency' : [10,100], # a list of relative carry frequencies to perform the carry on each weight, starting from the second highest order bit to the lowest
                                    # the number of carries should be equal to the number of cores_per_weight minus 1
    'exact_carries' : False, # should the carries be computed exactly (no read noise / nonlinearity etc).  Implies either write verify or digital carries
    'zero_reset' : ZeroResetPCEnum.CALIBRATED, # How to reset the ReRAMs after a carry:
                                          # "EXACT" the ReRAMs are reset to zero exactly (implies write verify) after a carry
                                          # "CALIBRATED"  use calibrated updates based on the current state,  i.e. write noise only, no nonlinearity (only implemented w/ lookup table model)
                                          # "BLIND"  use blind updates based on the average response, does not work well
    'min_carry_update' : None # if set to a number between 0 and 1, discretize carry updates to this fraction of the weight range.  Adjusts A/D precision and row/col update values to achieve discritization
}


ConvolutionDefaults = {
    'is_conv_core' : False, # whether the layer is a convolution layer
    'sameConv' : False, # whether to pad the input so that the input and output channels have the same size
    'px_0' : 0, # amount of padding in x on left
    'px_1' : 0, # amount of padding in x on right
    'py_0' : 0, # amount of padding in y on top
    'py_1' : 0, # amount of padding in y on bottom
    'stride' : 1, # stride of the convolution filter
    'Kx' : 3, # convolution filter size x
    'Ky' : 3, # convolution filter size y
    'Noc' : 1, # number of output channels
    'Nix' : 8, # x dimension input
    'Niy' : 8, # y dimension input
    'Nic' : 1, # number of input channels
    'bias' : False, # whether to have a row of bias weights
    'depthwise' : False, # Depthwise convolution
    'Nwindows'  : 0, # Total number of sliding windows per convolution in this layer
    'subarray_id' : 0, # if matrix is split, which partition the array corresponds to (do not set manually, set in convolution_core)
    'last_subarray' : True, # if matrix is split, whether partition is the final one
}

AnalyticsDefaults = {
    'store_weights' : False, #store the weights after each update
    'store_update_inputs' : False, #store the inputs for each update
    'max_storage_cycles' : 10000, # number of cycles for which to store internal results (to prevent memory cost from blowing up), 0 means store all
    'all_weights' : True , # if true, store all weights, if false, store weights specified by weight_rows and weight _cols
    'weight_rows' : (0,1,2), #row index of weights to save
    "weight_cols" : (0,1,2), # col index of weights to save
    "no_update_rows" : False, #don't store the row inputs for each update
    "no_update_cols" : False, #don't store the col inputs for each update
    "store_row_inputs" : False, # store all row inputs
    "store_col_inputs" : False, # store all col inputs
    "store_row_outputs" : False, # store all row outputs
    "store_col_outputs" : False, # store all col outputs
    "record_updates" : False, # record target and real updates
    "Nupdates_total" : 1e10, # total expected weight updates in first core (summed over all epochs)
}



ClipConstraintsDefaults = {
    'minimum' : -1.0,
    'maximum' :1.0,
}

# zero sigma means no noise
NormalErrorDefaults = {
    'sigma' : 0.0,  # sigma is defined relative to the full range
    'proportional' : False,
    # False: new value = old value + N(sigma*range)
    # True: new value = old value * [ 1+N(sigma) ]

    'keep_within_range' : True,  #clips the resulting value to the clipping constraints
    'randMat_pregen' : None, # Matrix of random numbers to be pre-generated
    'rowShuffle' : True # State variable keeping track of which dimension is being shuffled
}

UniformErrorDefaults = {
    'range' : 0.0,  #the range of the noise in fraction of whole range
    'keep_within_range' : True, #clips the resulting value to the clipping constraints
}

#zero bits means no quantization
QuantizationConstaintsDefaults = {
    'bits':  0,
    'sign_bit' : True,
    'stochastic_rounding' : False,
    'extra_half_bit' : False
}



# noise and quantization are only done on the inner core
# for weights, noise/quantization is only applied during updates.  Clipping is applied when setting the matrix.
XbarParamsDefaults = {
    'weights' : {
        'minimum' : 0.1,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False
        },
    'weight_clipping' : {
        'minimum' : 0.1,
        'maximum' : 1.0,
        },    # a separate clipping range is used for the weights so that the weights can be scaled to a target range,
              # but the models are defined outside of the target range
              # (only used for numeric simulations, xyce has it's own built in clipping range)
    'row_input' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False,
    },
    'col_input' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False,
    },
    'row_update' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False
    },
    'col_update' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False
    },
    'row_output' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False
    },
    'col_output' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
        'bits' : 0,
        'sign_bit' : True,
        'stochastic_rounding' : False
    },
    'attributes' : {
        'balanced_style' : 'two_sided', # one-sided means one of the devices is always at the lowest conductance, two_sided means the sum of the two conductances is fixed
        'NrowsMax' : 0,
        'input_bitslicing' : False, # Whether input is applied in bit serial fashion
        'Nimages_bitslicing' : 1, # Number of images to collect statistics from for bit slicing profiling (used to conserve memory)
        'interleaved_posneg' : False, # Whether positive and negative weight columns are interleaved in the same array to reduce parasitic voltage drops
        'Nslices' : 1, # number of weight bit slices
        'Nbits_reduction' : None, # number of bits by which to reduce the ADC range of the top bit slice
        'profile_ADC_inputs' : False, # profile pre-ADC input values; the format of output will depend on array size, input bitslicing, weight bitslicing
        'balanced_bitsliced' : False, # Used balanced core for each bit slice (implemented within BitSlicedCore)
        'fast_balanced' : False, # Implement MVM in BitSlicedCore rather than calling the method in NumericCore for speed
        'noRpSlices' : [0,0], # Whether Rp=0 for specific slices (LSB to MSB, 0 means Rp is enabled) 
        'ADC_per_ibit' : False, # ADC conversion after every input bit
        'digital_offset' : False, # Used in Offset Core only, whether offset is subtracted purely in digital (if False, uses unit column)
        'offset_inference' : False, # Tells Offset core to use inference mode (slightly different model)
        'clip_Icol' : False, # Clip column current before the ADC (models max current of analog integrator)
        'Icol_max' : 1, # Column current max when clip_Icol is True
        'adc_range_option' : 'calibrated', # Option for how ADC range is set: calibrated (user supplied), FPG, or max
        'adc_range_internal' : None, # ADC min/max if ADC_per_ibit is enabled. Stored here since it is set directly in xbar units, not algorithm
    }#any extra variables to be set
}

# parameters for numeric simulations
NumericParamDefaults = {
    'read_noise' : { #instance of normal error, only applied during VMM and MVM
        'sigma' : 0.0,
        'proportional' : False,
        'keep_within_range' : True,  #clips the resulting value to the clipping constraints
        'pregenRandom': False,
    },
    'write_noise' : {  # the write noise is set to zero if the update is zero

        'write_noise_model' : WriteNoiseModelEnum.G_INDEPENDENT,  # the type of write noise model (defined in enums section)
        'sigma' : 0 # noise sigma (dimensionless, normalized to range)

        # #result is automatically clipped to the clipping constraints

    },
    'nonlinearity' : {
        'alpha' : 0.0,  # zero means no nonlinearity, larger number is worse nonlinearity
        'symmetric' : False, # symmetric or asymmetric nonlinearity
        # TODO: Add option for offset zero reference if nonlinearity causes zero to be offset from the center
        # if ideal range is -10 to 10, a 10% zero shift may result in actual range of -8 to 12
        # print a warning with the actual range if zero is shifted

    },
    'dG_lookup' : {
        'file_increasing' : "TaOx",  # the location of the dG lookup table file for increasing pulses
        'file_decreasing' : "TaOx",  # the location of the dG lookup table file for decreasing pulses
        'Gmin_relative' : 0.1, # the starting point of the used conductance range (normalized to the range)
        'Gmax_relative' :0.9, # the starting point of the used conductance range (normalized to the range)
        'disable_nonlinearity' :False, # if true, the nonlinearity is ignored and only the write noise is set.  Can be changed dynamically for calibrated updates
        'disable_writenoise' : False, # if true, the write nosie is ignored and only the nonlinearity is used.  Can be changed dynamically
        'multi_LUT' : False, # whether multiple lookup table (device to device variability) is used
        'Nluts' : 1, # number of lookup tables (for device to device variability)
        'stochastic_quantization' : False, # whether stochastic rounding is used for quantization of conductance after update
        'Nlevels' : 0, # number of conductance levels to quantize to; 0 for continuous
        'w_limits' : None, # conductance limits between which to quantize weights to for each device LUT
        'gpu_engaged' : False,
        'vectorize' : False, # use vectorized LUT interpolation code
        'table_dim' : 1, # dimension of LUT
    },

    'circuit' : { # parameters used to simulate write circuitry
        'VrowS' : 1.0, # Selected row voltage
        'VrowUS' : -0.3, # Unselected row voltage
        'VcolUS' : 0.5, # Unselected column voltage
        'Vselect' : 0.6667, # Selector turn-on voltage
        'Vprog' : 0.1333, # Voltage above which device is programmed
        'Vread' : 0.7667, # Read voltage using during MVM/VMM with temporal encoding; must be higher than Vselect!
        'noRowParasitics' : False, # If True, no parasitic resistance is assumed on the rows (e.g. gate-connected)
        'selectorRead' : False, # Whether or not select device is modeled during read; True for best accuracy, False for speed
    },

    'attributes' : {
        'update_model' : UpdateModelEnum.ANALYTIC,
        'useGPU' : False,
        'gpu_id' : 0,
        'x_par': 1, # Number of sliding window steps in the x direction to do in parallel on GPU
        'y_par': 1, # Number of sliding window steps in the y direction to do in parallel on GPU
        'weight_reorder' : False, # Whether to enable weight reordering in conductance matrix to eliminate some zeros and allow even more parallelism
        'useEINSUM': False,
        'Rp': 0, # Normalized parasitic resistance of bit line
        'parasitic_backprop' : True, # Include parasitics in backprop and update?
        'Nex_par' : 1, # Number of examples to run in parallel when parasitics are enabled (only if GPU enabled)
        'Niters_max_parasitics' : 100, # Max number of iterations for parasitic circuit model (exceeding this causes model to conclude Rp is too high)
        'Verr_th_mvm' : 1e-3, # MVM/VMM error threshold for convergence in parasitic circuit model [Volts]
        'Verr_th_opu' : 1e-4, # OPU error threshold for convergence in parasitic circuit model [Volts]
        'convergence_param' : 1, # Convergence parameter for parasitic simulations
        'convergence_param_opu' : 1, # Convergence parameter for parasitic simulations
        'binary_update' : False, # Whether OPU updates are binarized
        'binary_threshold' : 0.5, # Threshold of update binarizing function
    }#any extra variables to be set
}

# parameters for weight programming errors and drift
WeightErrorParamsDefaults = {
    'T' : 0, # drift time (since last write-verify) assumed at inference
    'keep_within_range' : True, # clips the resulting value to the clipping constraints
    'sigma_error' : 0, # Error spread to be applied in the case of weight-independent programming error
    'proportional' : False, # Whether random programming error is applied proportionally (False: additively)
    'error_model' : "none", # How programming error is applied as a function of the weight value
    'noise_model' : "none", # How read noise is applied as a function of the weight value
    'drift_model' : "none", # How read noise is applied as a function of the weight value
    'TID' : 0, # Total ionizing dose for SONOS rad effects model
    'rad_type' : None, # TID drift model used
    'W0_vec' : None,  # list of weight values whose alpha values are known empirically (deprecated)
    'sigma0_vec' : None, # list of alpha values corresponding to W0_vec (deprecated)
}


AlgorithmParamsDefaults = {
    'weights' : {
        'minimum' : -10,
        'maximum' : 10,
        },
    'row_input' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
    },
    'col_input' : {
        'minimum' : -1.0,
        'maximum' : 1.0,
    },
    'row_update' : {
        'minimum' : -5,
        'maximum' : 5,
    },
    'col_update' : {
        'minimum' : -5,
        'maximum' : 5,
    },
    'row_output' : {
        'minimum' : -20,
        'maximum' : 20,
    },
    'col_output' : {
        'minimum' : -20,
        'maximum' : 20,
    },
    'attributes' : {
        'row_update_portion' : 1,  #0 to 1.  Should be 1 when row driver is timed in Xyce so that the learning rate scales the length of pulses.  Applying to level driver will fail for learning rates > 1 as V/3 write scheme breaks down
        'subtract_current_in_xbar': True,  #this should be true or else weird clipping will result
        'calculate_inner_from_outer': True,
        'disable_clipping' : False,
        'serial_read_scaling': 1, # when doing a serial read, the output of the integrator is scaled by this value before being passed to the A/D and then unscaled after A/D
        'crossbar_type' : CrossbarTypeEnum.OFFSET,
        'sim_type' : SimTypeEnum.NUMERIC,
        'weight_bits' : 0, # Weight quantization applied during import; 0 to disable
    }#any extra variables to be set
}


# nothing user defined here
WrapperParamsDefaults = {
    'attributes' : {
    }
}




# ************************************* XYCE PROPERTIES ***************************************************************



XyceParametersDefaults = {
    'xyce_path':"/usr/local/Xyce-DEVELOPMENT-7.0.0-NORAD/bin/Xyce",# '/usr/local/Xyce-DEVELOPMENT-7.0.0-OPENSOURCE/bin/runxyce',
    'xyce_parallel_path' : '/usr/local/Xyce-Release-6.5.0-OPENMPI-NORAD/bin/xmpirun',  #alternatively this can be set to an mpi command, ie.
                    # 'mpirun --hostfile ~/mpihosts.txt Xyce'  this assumes that PATH andd LD_LIBRARY_PATH is set correctly
    'n_processors' : 1,
    'xyce_run_command' : None, # if set to a value this command use used to launch xyce instead of the xyce path above + nprocs (used for mpirun)
                              # SECURITY WARNING:  Can run arbitrary code through this option

    'parallel_solve' : True, # if n_procs>1, use the parallel matrix solver in addition to the parallel load in xyce
    'xyce_options' : '',  # an arbitrary line added to the xyce simulation to customize the simulation, typically a .options statement
    'parallel_timeout': None, # None =  no timeout, otherwise set to a time in seconds after which a parallel simulation will be terminated a serial simulation run instead

    'debug_output': False,
    'calculate_power': False,
    'output_rram_voltages' : False,
    'parasitic_compensation' : ParasiticCompensationEnum.NONE, # what parasitic compensation scheme to use
    'parasitic_compensation_parallelism' : 1, # if using the partially parallel scheme, how many devices are activated at a time
    'xyce_time_stepping' : XyceTimeSteppingEnum.FULL, # how many time steps should xyce use
    'print_all_time_steps' : False,
    'out_dir' : None ,# output directory for all Xyce files, if none a temporary directory is used
    # 'del_out_dir' : False # DO Not Change, used internally to keep track of whether output directory should be deleted
}



# **************** shared driver parameters

#    A :py:class:`WriteParameters` instance for positive writes (attempting to provide a positive voltage across the memristor)
WritePosDefaults = {
    'pulse_max_width' :1e-6, # How long the write cycle lasts
}

#    A :py:class:`WriteParameters` instance for positive negative (attempting to provide a negative voltage across the memristor)
WriteNegDefaults = {
    'pulse_max_width' :1e-6, # How long the write cycle lasts
}


# A set of parameters that must be shared by both drivers (either specific to writing, or otherwise don't make sense to be allowed to be different for rows and columns)
# times apply to both the memory and neural circuits
SharedDriverParameterDefaults = {
    'initial_wait_time' : 10e-9,    #RTN needs time to initialize, so wait before starting any pulses    #An initial "settling" time for Xyce to wait before any drivers start (actually needed for RTN to initialize)
    'rise_time' : 1e-12, #Rise time for drivers' voltage source (note that RC delays could increase the actual value)
    'fall_time' :1e-12, # Fall time for drivers' voltage source (note that RC delays could increase the actual value)
    'multi_pulse_hold_time' : 10e-9, # time between multiple write pulse phases to hold voltage at zero, should be larger than rise time
    'simulation_settle_time': 10e-9,  # Amount of time to let the simulation settle after the read or write


    ##### The following are for neural circuits only
    'drive_impedance' :1.0, #    Output impedance for the driver when it is connected to the circuit (e.g.: pass transistor is in the low-resistance state)
    'highz_impedance' : 1e9,  #    Output impedance for the driver when it is disconnected from the circuit (e.g.: pass transistor is in the high-resistance state)


    # **** Read Parameters
    # using calibrated IBM Access device
    'read_voltage_pos' :1.3, #On/Off = 6.92, Vrram = 0.246
    'read_voltage_neg' :-1.3,  #TODO: calibrate negative voltage to  give same current as positive voltage
    'read_pulse_max_width': 100e-9, #  Maximum read pulse (in seconds)

    'calibrate_voltage_driver': False  # If False, the voltage driver is controlled by linearly interpolating between the max and min voltage, if True, the driver is calibrated based on the memristor/access device model, assuming no parasitics


}


# ********************************  driver specific parameters
# Driver-specific (row vs. column) parameters

# customize row vs col write drivers
XyceRowDriverDefaults = {
    'write_style': DriverStyleEnum.TIMED,  #The write style for this driver (must be unique among the two drivers)
    'read_style' :DriverStyleEnum.TIMED, #  Don't change, Type of read pulse used for the driver ONLY TIMED DRIVER IS IMPLEMENTED
}

XyceColDriverDefaults = {
    'write_style': DriverStyleEnum.LEVEL,
    'read_style' :DriverStyleEnum.TIMED, #  Don't change, Type of read pulse used for the driver ONLY TIMED DRIVER IS IMPLEMENTED
}


# ************************   neuron parameters
NeuronParameterDefaults = {
    'opamp_gain' :1e4,
    'load_resistance' : 1,  # the load resistance through which current is calculated for RESISTOR and VCVS-R neurons
    'integrator_capacitance' : 10e-12,
    'input_impedance' : 1.0,
    'highz_impedance' : 1e9,
    'style' : NeuronStyleEnum.VCVS_C
}



# **************************** device and access device model parameters

#             #These parameters should make the device linear  (add example params here)


# ****************************** parasitic properties ******************************


#ITRS 2013 M1 (14nm)
_half_pitch = 40e-9 #m
_wire_width = _half_pitch
_wire_height = _wire_width*1.9
_wire_resistivity = 4.03e-8 #ohms m

_wire_r = _wire_resistivity*1/(_wire_width*_wire_height) #13.26 e6 Ohm/m
_wire_cap  = 0.19e-9 # F/m

# # IBM 32 nm from alex
# # _half_pitch = 50e-9
# # _wire_r = 7.87e6  #Ohm/m
# # _wire_cap  = 0.2281e-9 # F/m
#
# _LINE_RESISTANCE = _wire_r*2*_half_pitch   #0.787 Ohms (IBM)  1.06(ITRS)
# _LINE_CAPACITANCE = _wire_cap*2*_half_pitch   #22.8 aF (IBM)  #15.2 aF (ITRS)



XyceXbarParameterDefaults = {
    'USE_PARASITICS' : True,  # use rram capacitance, line capacitance, line resistance, driver and neuron resistance
    'LUMPED_PARASITICS' : True, #runs faster than transmission line model

    'cell_spacing' : _half_pitch*2,  #m
    'cell_resistance' : _wire_r*_half_pitch*2,   #ohms
    'resistance_sigma' : 0, # the standard deviation of the parastic resistance, normalized to the cell_resistance
    'cell_capacitance' : _wire_cap*_half_pitch*2,   #F  #zero removes the capacitor
    'cell_inductance' : 0.0,    #L  # only used with transmission line model, if LUMPED_PARASITICS=False
}

# These parameters specify how specific devices (memristor + access device, if enabled) behave

DeviceParameterDefaults = {
    'USE_DEVICE_CAPACITANCE' : True, #if false, disable ReRAM and AD capacitance
    'GROUND_DEVICE_CAPACITANCE' : False,# if true, connects series combo of device + access dev cap to ground
                                        # (useful for allowing parallel sims to run)

    'USE_ACCESS_DEVICE' : True,#    Whether to include an access device
    'model' : DeviceModelEnum.YAKOPCIC,

    'Gmin_relative' : 0.1, # the starting point of the used conductance range (normalized to the range)
    'Gmax_relative' :0.9, # the ending point of the used conductance range (normalized to the range)

    # Narayanan et all Exploring the design space for crossbar arrays built with MIEC Access Devices
    'ACCESS_DEVICE_V0' :  (85e-3*math.log10(math.e)),
    # define IO in terms of the voltage margin at 10 nA
    'ACCESS_DEVICE_voltage_margin' : 1.54,
        # ACCESS_DEVICE_I0 = 10e-9/(math.exp(self.ACCESS_DEVICE_voltage_margin/(2*self.ACCESS_DEVICE_V0))-1)
    'access_device_capacitance' : 8.854e-12*25*_half_pitch**2/10e-9,  #set to be same as ReRAM cap

    'rram_capacitance' : 8.854e-12*25*_half_pitch**2/10e-9,  #55.3 aF (for IBM)   #35.4 aF (for ITRS)


    # ****************************** Piecewise Empirical Model ****************************
    'PEM_fxpdata' : None, #location of a file specifying f(x) for positive pulses,
                          # if None, the average response of the lookup table specified by dG_lookup is used\
                            #f(x) will be normalized so that the average is 1 in the range specified by dGlookup params
    'PEM_fxmdata' : None, #location of a file specifying f(x) for negative pulses,
                          # if None, the average response of the lookuptable specified by dG_lookup is used
    'PEM_I1' : 10e-6,
    'PEM_I2' : 10e-6,
    'PEM_V1' : 0.265,
    'PEM_V2' : 0.265,
    'PEM_G0' : 130e-6,
    'PEM_VP' : 0.5,
    'PEM_VN' : -0.5,
    'PEM_d1' : 6,
    'PEM_d2' : -6,
    'PEM_C1' : 1e4,
    'PEM_C2' : 1e4,



    # ****************************** YAKOPCIC model parameters  ******************************

    # data loosely calibrated to david's measurements
    'YAKOPCIC_A1' : 3.8e-4,  #Ron = 1.2 kOhms
    'YAKOPCIC_A2' : 3.8e-4,    #Ron = 1.2 kOhms
    'YAKOPCIC_B' : 2.2,

    #Write at V=0.3.
    'YAKOPCIC_VP' : 0.3,
    'YAKOPCIC_VN' : 0.3,
    # #Vp. Vn chosen large to surpress all writing
    #     YAKOPCIC_VP' : 0.9,
    #     YAKOPCIC_VN' : 0.9,

    'YAKOPCIC_AP' : 1e5,
    'YAKOPCIC_AN' : 1e5,

    #     YAKOPCIC_XP' : 0.95,  # make linear
    #     YAKOPCIC_XN' : 0.95,  # make linear
    'YAKOPCIC_XP' : 0.2, # strong nonlinearity
    'YAKOPCIC_XN' : 0.2,
    'YAKOPCIC_ALPHAP' : 1,
    'YAKOPCIC_ALPHAN' : 5,

    'YAKOPCIC_ETA' : 1,


    #############
    'RESNOISE' : False,
    'RESSEED' : 100,
    'RESLAMBDA' : 1,
    'RESTD' : 1e-9,  # average time delay between RTN events. In practice actual delay is reslambda * restd
    'RESEPTD' : 1e-12,
    'RESDELTA' : 2,
    'RESDELTAGRAD' : 0.01
    # """
    # Parameter   Description                                 Units   Default
    #
    # resnoise    true=use RTN model, false=no RTN model              false
    # resseed     integer seed>
    # reslambda   mean of poisson distribution used in RTN model
    # restd       average time delay between RTN events
    #             In practice actual delay is reslambda * restd
    # reseptd     smallest allowed time delay
    #             in case a time delay of 0 is bad
    # resdelta    change in resistance due to RTN             -
    # resdeltagrad change in resistance due to RTN scaled     1/ohm
    #             by actual resistance
    # """
}

# ************************   memory parameters  (may be dynamically changed)
# use None for a read voltage to represent a High Z impedance connected to ground

MemoryParameterDefaults = {
    'Ileak' : 1e-6,
    'read_model' : MemoryReadModelEnum.ALL_GND,
    'write_time' : 1e-6,
    'read_time' : 10e-9,

    'Vword_read' : 0.5,
    'Vbit_read' : -0.5,
    'Vrow_read' : 0,
    'Vcol_read' : 0,

    'Vword_write' : 2,
    'Vbit_write' : 0,
    'Vrow_write' : 1,
    'Vcol_write' : 1,

    'drive_impedance' : 1,
    'read_impedance' : 1e4,
    'highz_impedance' : 1e9

}
