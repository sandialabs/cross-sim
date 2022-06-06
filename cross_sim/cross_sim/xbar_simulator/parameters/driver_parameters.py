#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#



from .base import ParametersBase, Parameter
from  . import parameter_defaults as params
from .parameter_defaults import DriverStyleEnum

class WriteParameters(ParametersBase):
    '''
    A set of write parameters (when writing either positive values or negative values)
    Is a a subset of SharedDriverParameters
    '''

    if False:
        pulse_max_width = float

    # set post_set function
    voltage_min = Parameter(name="voltage_min", post_set=None)
    voltage_max = Parameter(name="voltage_max", post_set=None)

    def __init__(self, param_root, **kwargs):

        # parameters must be passed from the calling function SharedDriverParameters
        ParametersBase.__init__(self,param_root, **kwargs)



class SharedDriverParameters(ParametersBase):
    '''
    A set of parameters that must be shared by both drivers (either specific to writing, or otherwise don't make sense to be allowed to be different for rows and columns)
    '''

    # define var names for easy code completion (these can be commented out, it's only to help the IDE)
    if False:
        write_pos = WriteParameters
        write_neg = WriteParameters
        initial_wait_time = float    #RTN needs time to initialize, so wait before starting any pulses    #An initial "settling" time for Xyce to wait before any drivers start (actually needed for RTN to initialize)
        rise_time = float  #Rise time for drivers' voltage source (note that RC delays could increase the actual value)
        fall_time = float # Fall time for drivers' voltage source (note that RC delays could increase the actual value)
        multi_pulse_hold_time = float # time between multiple write pulse phases to hold voltage at zero
        simulation_settle_time = float

        read_voltage_pos = float
        read_voltage_neg = float
        read_pulse_max_width = float #  Maximum read pulse (in seconds)
        drive_impedance = float #    Output impedance for the driver when it is connected to the circuit (e.g.: pass transistor is in the low-resistance state)
        highz_impedance = float  #    Output impedance for the driver when it is disconnected from the circuit (e.g.: pass transistor is in the high-resistance state)
        calibrate_voltage_driver=bool


    def __init__(self, param_root):
        attributes = params.SharedDriverParameterDefaults.copy()

        attributes['write_pos']=WriteParameters(param_root,**params.WritePosDefaults)
        '''
        A :py:class:`WriteParameters` instance for positive writes (attempting to provide a positive voltage across the memristor)
        '''
        attributes['write_neg']=WriteParameters(param_root,**params.WriteNegDefaults)
        '''
        A :py:class:`WriteParameters` instance for positive negative (attempting to provide a negative voltage across the memristor)
        '''

        #load the defaults
        ParametersBase.__init__(self,param_root, **attributes)


    @property
    def write_pulse_max_width(self):
        '''
        The maximum width of any write pulse
        '''
        return max(self.write_pos.pulse_max_width, self.write_neg.pulse_max_width)
    
    @property
    def write_pulse_max_period(self):
        '''
        The maximum width of any write pulse, including rise and fall time
        '''
        return self.write_pulse_max_width + self.rise_time + self.fall_time

    def write_sim_time_needed(self, duration_scale_factor=1):
        '''
        How long is actually required for a simulated write in a neural core
        '''
        return (
                self.initial_wait_time +
                ((self.rise_time+self.fall_time) * 4) +
                (self.write_pos.pulse_max_width * 2 *duration_scale_factor) +
                (self.write_neg.pulse_max_width * 2 *duration_scale_factor) +
                +self.multi_pulse_hold_time*4+self.simulation_settle_time
                )

    @property
    def read_sim_time_needed(self):
        '''
        How long Xyce needs to run for this read
        '''
        return self.initial_wait_time + self.read_pulse_max_period + self.simulation_settle_time


    @property
    def read_pulse_max_period(self):
        '''
        The longest possible time that the driver would have non-zero output during a read
        '''
        return self.read_pulse_max_width + self.rise_time + self.fall_time



class DriverParameters(ParametersBase):
    '''
    Driver-specific (row vs. column) parameters (at least those parameters for which it makes sense that row and column drivers could have different values)
    '''


    def __init__(self, param_root, **kwargs):
        # attributes = params.DriverParameterDefaults.copy()
        # attributes.update(kwargs)
        #load the defaults
        ParametersBase.__init__(self,param_root, **kwargs)

    write_style = Parameter(name="write_style",
                            post_set=ParametersBase.generate_enum_post_set("write_style", DriverStyleEnum))
    read_style = Parameter(name="read_style",
                           post_set=ParametersBase.generate_enum_post_set("read_style", DriverStyleEnum))
