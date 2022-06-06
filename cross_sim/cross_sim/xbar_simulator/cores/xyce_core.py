#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from tempfile import mkdtemp
import shutil
import os
from itertools import chain
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import interp1d

from .debug import print_debug_calls, DEBUG_CALLS
from ..xyce_blocks import Xbar, Driver, Neuron
from .clipper_core import ClipperCore #, PostprocUnsupported
from ..parameters import NeuronStyleEnum
from ..parameters import ParasiticCompensationEnum, XyceTimeSteppingEnum
from ..parameters.parameter_defaults import DriverStyleEnum
from warnings import warn


DEBUG_NUMERIC = False

DEBUG_NUMERIC = DEBUG_NUMERIC and DEBUG_CALLS

class XyceData(object):
    '''
    Column-based access to a Xyce output file (.prn)
    '''
    
    def __init__(self, path):
        '''
        :param path: Path to a Xyce print (.prn) file
        :type path: str
        '''

    # check numpy version
        from distutils.version import LooseVersion
        if LooseVersion(np.__version__) < LooseVersion('1.5'):
            #Why the heck are you using such an old version?
            self.data = np.genfromtxt(path,
                                  skiprows=1, skip_footer=1)
        else:
            #skiprows is now skip_header (since 1.5)
            self.data = np.genfromtxt(path, skip_header=1, skip_footer=1)
        
        self.next_col = 0
    
    def get_data(self, cols = 1, rowspec = slice(None, None, None)):
        '''
        Grab a requested number of columns from the output file; optionally specifying which rows are wanted (default is all rows).
        Often the number of cols needed based on the size of the xbar.
        subsequent calls to this will return the subsequent columns that have not yet been returned.
        Allows the data to be sequentially obtained.
        '''
        data = self.data[rowspec,self.next_col:self.next_col+cols]
        self.next_col += cols
        return data
        

class XyceCore(ClipperCore):

    core_count=0

    def __make_filename(self, filename):
        return os.path.join(self.dirname, filename)

    def __init__(self, params):
        """

        :param params: all parameters
        :type params: Parameters
        :param dirname:
        :return:
        """

        self.xyce_params = params.xyce_parameters

        XyceCore.core_count= XyceCore.core_count+1
        self.dirname = os.path.join(self.xyce_params.out_dir,"core"+str(self.core_count))
        os.makedirs(self.dirname, exist_ok=True)

        # # initialize output directory
        # if dirname is None:
        #     self.dirname = mkdtemp()
        #     self.del_dir = True
        # else:
        #     self.dirname = dirname
        #     self.del_dir = False
        #
        # if not os.path.exists(self.dirname):
        #     os.makedirs(self.dirname, exist_ok=True)
        

        self.xyce_state_vars = None  # the internal state variables for xyce

        #intitalize a 1x1 crossbar
        # the crossbar size is initialized when set_matrix is called.
        # the set numeric weights are converted to xyce internal state vars
        self.xbar = Xbar(1, 1, self.xyce_params.xbar)
        """:type: Xbar"""
        self.xbar_name = 'CrossBar'

        # initialize drivers and neuron objects
        self.row_driver = Driver(self.xyce_params.row_driver,self.xbar, '_row')
        self.col_driver = Driver(self.xyce_params.col_driver,self.xbar,'_col')
        self.row_neuron = Neuron(self.xyce_params.row_neuron,'_row')
        self.col_neuron = Neuron(self.xyce_params.col_neuron,'_col')

        # initialize driver parasitics as np arrays
        self.row_parasitic_scale = np.ones(1)
        self.col_parasitic_scale = np.ones(1)

        # initialize rows and cols to be printed to none, (set to list of tuples to print voltages at particular devices
        self.print_locations = None

        # compute initial conditions (need to calculate when initializing memory write to nonzero voltage)
        self.calculate_initial_conditions = False

        #define all the filenames

        # TODO: filenames can't represent the circuit properties until the # rows/cols are defined
        # self.crosspoint_file = self.__make_filename('crosspoint_DEVICE.cir')
        # self.xbar_file = self.__make_filename('xbarROWSxCOLS_DEVICE.cir')
        # self.xyce_file = self.__make_filename('xyce_DEVICE_xbarROWS_COLS.cir')

        self.crosspoint_file = self.__make_filename('crosspoint.cir')
        self.xbar_file = self.__make_filename('xbar.cir')
        self.xyce_file = self.__make_filename('xyce.cir')


        self.driver_file = self.__make_filename('driver.cir')
        self.neuron_file = self.__make_filename('neuron.cir')
        self.weights_file = self.__make_filename('set_weights.cir')
        self.screen_file = self.xyce_file + '.out'
        self.out_file = self.xyce_file + '.prn'
        
        ClipperCore.__init__(self, params)

        ## calculate and set the write voltages
        Vmin_neg, Vmax_neg, Vmin_pos, Vmax_pos = self.xbar.crosspoint.device.compute_write_voltage_limits()

        print("The ideal min/max write voltages are",Vmin_neg, Vmax_neg, Vmin_pos, Vmax_pos)
        if Vmax_neg<Vmin_neg*3 or Vmax_pos>Vmin_pos*3:
            warn("The max write voltage is >3X the min voltage and will cause a write disturb")
        params.xyce_parameters.shared_driver.write_pos.voltage_min  = Vmin_pos
        params.xyce_parameters.shared_driver.write_pos.voltage_max  = Vmax_pos

        params.xyce_parameters.shared_driver.write_neg.voltage_min  = Vmin_neg
        params.xyce_parameters.shared_driver.write_neg.voltage_max  = Vmax_neg

    
    # def __del__(self):
    #     '''
    #     If we made up a name for the temporary directory, delete it as we're dying
    #     this is called when the object is destroyed
    #     '''
    #
    #     if self.del_dir:
    #         if os.path.exists(self.dirname):
    #             shutil.rmtree(self.dirname)

    # define lists of node names
    # chose to use a generator rather than returning a list
    # the edges of the crossbar are defined as:
    #   left edge: node_rXX_left
    #   right edge: node_rXX_right
    #   top edge: node_cXX_top
    #   bottom edge: node_cXX_bottom

    def __make_rows_left(self):
        #return ('node_r{0}_left'.format(row) for row in range(self.rows))
        for row in range(self.rows):
            yield 'node_r{0}_c0_left'.format(row)
            
    def __make_rows_right(self):
        for row in range(self.rows):
            yield 'node_r{0}_c{1}_right'.format(row,self.cols-1)
    
    def __make_cols_top(self):
        for col in range(self.cols):
            yield 'node_r0_c{0}_top'.format(col)
        
    def __make_cols_bottom(self):
        for col in range(self.cols):
            yield 'node_r{0}_c{1}_bottom'.format(self.rows-1,col)
    
    def __make_xbar_nodes(self):
        return chain(self.__make_rows_left(), self.__make_rows_right(), self.__make_cols_top(), self.__make_cols_bottom())
        # chains nodes one after the other (equivalent to appending each list one after the other)


    # define the lists of output nodes after the neuron/opamp circuit
    #   row neuron outputs:  output_rXX
    #   col neuron outputs:  output_cXX

    def __make_rows_neuron_output(self):
        for row in range(self.rows):
            yield 'output_r{0}'.format(row)

    def __make_cols_neuron_output(self):
        for col in range(self.cols):
            yield 'output_c{0}'.format(col)

    # define the subcircuit names of all the drivers and neurons
    #   row drivers:  Xdriver_rXX
    #   col drivers:  Xdriver_cXX
    #   row neurons:  Xneuron_rXX
    #   col neurons:  Xneuron_cXX
    def __make_row_driver_name(self):
        for row in range(self.rows):
            yield 'Xdriver_r{0}'.format(row)
                
    def __make_col_driver_name(self):
        for col in range(self.cols):
            yield 'Xdriver_c{0}'.format(col)
                
    def __make_row_neuron_name(self):
        for row in range(self.rows):
            yield 'Xneuron_r{0}'.format(row)
                
    def __make_col_neuron_name(self):
        for col in range(self.cols):
            yield 'Xneuron_c{0}'.format(col)

    # create parasitic compensation resistances
    def _driver_resistances(self, is_row, is_output=False, is_read=False):
        """
        return resistances needed for parasitic compensation

        :param is_row: generate resistances for rows or cols
        :param is_output: transpose the resistances for output edges bottom / right
        :return:
        """
        if is_row:
            ndrivers= self.rows
        else:
            ndrivers=self.cols

        cell_resistance=self.params.xyce_parameters.xbar.cell_resistance

        #no compensation
        if self.params.xyce_parameters.parasitic_compensation == ParasiticCompensationEnum.NONE:
            return np.zeros(ndrivers)

        #serial read/write (i.e. memory)
        elif self.params.xyce_parameters.parasitic_compensation == ParasiticCompensationEnum.SERIAL:
            resistances= cell_resistance*np.arange(ndrivers, dtype=np.float)

        # parallel read/write (i.e. neural)
        elif self.params.xyce_parameters.parasitic_compensation == ParasiticCompensationEnum.PARALLEL:
            resistances = np.arange(ndrivers, dtype=np.float)
            resistances*=resistances+1
            resistances *=cell_resistance/2/ndrivers
            #cell_resisitance*n*(n+1)/(2*n_max)  (series sum is n*(n+1)/2, but current is ndrivers times larger)

        # partially parallel read for memory
        elif self.params.xyce_parameters.parasitic_compensation == ParasiticCompensationEnum.PARTIALLY_PARALLEL:
            resistances = cell_resistance*np.arange(ndrivers, dtype=np.float)
            if is_output:  # column resistances need to be multiplied by parallelism as current through a single row is split over multiple columns
                resistances*=self.params.xyce_parameters.parasitic_compensation_parallelism
        else:
            raise ValueError("Undefined parasitic compensations scheme: "+str(self.params.xyce_parameters.parasitic_compensation))

        # add variability
        resistance_sigma = self.xyce_params.xbar.resistance_sigma
        if resistance_sigma:
            # load saved parasitic scales if possible:
            if is_row:
                if self.row_parasitic_scale.size ==ndrivers:
                    parasitic_scale=self.row_parasitic_scale
                else:
                    parasitic_scale=np.random.normal(size=ndrivers)
                    self.row_parasitic_scale = parasitic_scale
            else:
                if self.col_parasitic_scale.size ==ndrivers:
                    parasitic_scale=self.col_parasitic_scale
                else:
                    parasitic_scale=np.random.normal(size=ndrivers)
                    self.col_parasitic_scale = parasitic_scale


            #add noise proportional to sqrt of the size
            size = resistances/cell_resistance
            resistances+= np.sqrt(size)*(resistance_sigma*cell_resistance)*parasitic_scale
            resistances[resistances<0]=0


        # transpose resistances as needed
        if is_read:
            if is_output:
                return resistances[::-1]
            else:
                return resistances
        else: # is write (only done from top left)
                return resistances[::-1]

    def _driver_capacitances(self, is_row):
        raise NotImplemented


    def __set_weights(self):
        '''
        Write the internal weights to the weights file
        '''
        with open(self.weights_file, 'w') as fs:
            self.xbar.SetInternalVars(fs, 'XCrossBar1', self.xyce_state_vars)

    def set_matrix(self, matrix):
        """
        initializes the xbar object, writes the xbar circuit, calculates and stores the internal weights
        :param matrix:
        :return:
        """
        matrix = self.clip_matrix(matrix)

        # reset the crossbar size
        self.xbar.rows=self.rows
        self.xbar.cols=self.cols

        with open(self.xbar_file, 'w') as fs:
            self.xbar.WriteSubckt(fs)
        
        self.xyce_state_vars = self.xbar.crosspoint.device.CalculateXyceStateVar(matrix.copy())

    
    def set_vmm_inputs(self, vector):
        vector = self.clip_vmm_inputs(vector)
        if vector.size != self.rows:
            raise ValueError('Input vector size ({0}) should be the same as the number of rows ({1}).'.format(vector.size, self.rows))
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        # clip and quantize vector
        vector = self.clip_mvm_inputs(vector)

        if vector.size != self.cols:
            raise ValueError('Input vector size ({0}) should be the same as the number of cols ({1}).'.format(vector.size, self.cols))
        self.vector_mvm = vector
        
    
    def __run(self, is_row_read = False, is_col_read = False, is_write = False, is_memory =False):
        """
        Do final file writing, run Xyce, and read back the results

        :param is_row_read: is this a VMM, if so output col neuron voltage & row drivers (also used for row input memory)
        :param is_col_read: is this a MVM, if so output row neuron voltage & col drivers (also used for col input memory)
        :param is_write: set sim length based on whether it is a write or read
        :param is_memory:  set sim length based on whether is a memory crossbar or not
        :return:
        """
        self.__set_weights()    #Do just before running, so we keep previous internal var file around

        with open(self.xyce_file, 'w') as ostream:
            ostream.write('*\n\n.include "{0}"\n\n'.format(self.xbar_file))
            self.xbar.WriteCrossbar(ostream, 'XCrossBar1',self.__make_xbar_nodes())
            
            ostream.write('* driver conditions\n.include "{0}"\n\n'.format(self.driver_file))
            
            ostream.write('* neurons\n.include "{0}"\n\n'.format(self.neuron_file))
            
            ostream.write('* weight settings\n.include "{0}"\n\n'.format(self.weights_file))


            # ********************  create preprocess statement to remove dangling resistors
            ostream.write('.PREPROCESS REMOVEUNUSED R,C\n\n')

            ostream.write('*Use MIN in measure statements to exclude energy fed back to the driver\n')
            # ******************** create measure statements to calculate the energy
            if self.xyce_params.debug_output or self.xyce_params.calculate_power:
                # integrate V*I for cols
                if is_col_read or is_write:
                    for n in self.__make_col_driver_name():
                        ostream.write('.MEASURE TRAN energy_{0} INTEGRAL {{MIN(v({0}:temp)*i({0}:Vdrv),0)}} \n'.format(n))

                # integrate V*I for rows
                if is_row_read or is_write:
                    for n in self.__make_row_driver_name():
                        ostream.write('.MEASURE TRAN energy_{0} INTEGRAL {{MIN(v({0}:temp)*i({0}:Vdrv),0)}} \n'.format(n))
            #TODO:  Add measurement for read neurons


            # ****************** create print statement to print all outputs (order matters for processing output file)
            ostream.write('\n* output\n')
            ostream.write('.print tran FORMAT=NOINDEX\n')

            # set precision to 4 to limit output file size
            # ostream.write('.print tran FORMAT=NOINDEX PRECISION=4\n')

            if is_col_read:
                for n in self.__make_rows_neuron_output():
                    # print row neuron output voltage
                    ostream.write('+ v({0})\n'.format(n))
            if is_row_read:
                for n in self.__make_cols_neuron_output():
                    #print col neuron output voltage
                    ostream.write('+ v({0})\n'.format(n))

            self.xbar.PrintInternalVars(ostream, 'XCrossBar1')
            
            # print driver currents and voltages for debugging and calculating driver_power.
            # can only integrate driver currents if debug_output is true, or else too few data points will be output (use internal measure statements instead)
            if self.xyce_params.debug_output or self.xyce_params.calculate_power:
                # print col driver voltages and currents
                if is_col_read or is_write:
                    for n in self.__make_cols_top():
                        ostream.write('+ v({0})\n'.format(n))
                    for n in self.__make_col_driver_name():
                        ostream.write('+ v({0}:temp)\n'.format(n))
                    for n in self.__make_col_driver_name():
                        ostream.write('+ i({0}:Vdrv)\n'.format(n))
                    # print calculated energy
                    for n in self.__make_col_driver_name():
                        ostream.write('+ energy_{0}\n'.format(n))

                # print row driver voltages and currents
                if is_row_read or is_write:
                    for n in self.__make_rows_left():
                        ostream.write('+ v({0})\n'.format(n))
                    for n in self.__make_row_driver_name():
                        ostream.write('+ v({0}:temp)\n'.format(n))
                    for n in self.__make_row_driver_name():
                        ostream.write('+ i({0}:Vdrv)\n'.format(n))
                    # print calculated energy
                    for n in self.__make_row_driver_name():
                        ostream.write('+ energy_{0}\n'.format(n))

            if self.xyce_params.debug_output:
                if is_col_read and self.xyce_params.row_neuron.style==NeuronStyleEnum.VCVS_C and not is_memory:
                    #TODO: only works with capacitance integrating neuron
                    for n in self.__make_row_neuron_name():
                        ostream.write('+ i({0}:Cint)\n'.format(n))
                if is_row_read and self.xyce_params.col_neuron.style==NeuronStyleEnum.VCVS_C and not is_memory:
                    for n in self.__make_col_neuron_name():
                        ostream.write('+ i({0}:Cint)\n'.format(n))
                
                self.xbar.PrintInternalResistances(ostream, 'XCrossBar1')
                self.xbar.PrintInternalCurrents(ostream, 'XCrossBar1')
                self.xbar.PrintAccessDeviceVoltages(ostream, 'XCrossBar1')
                self.xbar.PrintRowVoltages(ostream, 'XCrossBar1')
                # self.xbar.PrintColVoltages(ostream, 'XCrossBar1')

            if self.xyce_params.debug_output or self.xyce_params.output_rram_voltages:
                self.xbar.PrintInternalVoltages(ostream, 'XCrossBar1')

            # print specific xbar locations
            if self.print_locations is not None:
                self.xbar.PrintInternalVoltages(ostream, 'XCrossBar1',print_locations=self.print_locations)
                self.xbar.PrintAccessDeviceVoltages(ostream, 'XCrossBar1',print_locations=self.print_locations)



            # ************************* create .tran statement to run the simulation

            # set simulation time based on simulation being run
            if is_memory:
                shared_driver = self.xyce_params.shared_driver
                if is_write:
                    time_needed = (shared_driver.initial_wait_time +
                        shared_driver.rise_time+shared_driver.fall_time + self.params.memory_params.write_time+
                        self.xyce_params.shared_driver.simulation_settle_time)
                else: # is read
                    time_needed = (shared_driver.initial_wait_time +
                        shared_driver.rise_time+shared_driver.fall_time + self.params.memory_params.read_time+
                        self.xyce_params.shared_driver.simulation_settle_time)
            else: # is neural xbar
                if is_write:
                    time_needed = self.xyce_params.shared_driver.write_sim_time_needed(duration_scale_factor=self.row_driver.duration_scale_factor)
                else:
                    time_needed = self.xyce_params.shared_driver.read_sim_time_needed
            
            ostream.write('\n\n* run simulation\n')

            # write different transient statements depending on how many time points should be solved for
            if self.params.xyce_parameters.xyce_time_stepping == XyceTimeSteppingEnum.FULL:
                if self.calculate_initial_conditions:
                    ostream.write('.tran 0 {0} \n\n'.format(time_needed))
                else:
                # setting UIC causes initial conditions to be set to zero
                    ostream.write('.tran 0 {0} UIC\n\n'.format(time_needed))

            elif self.params.xyce_parameters.xyce_time_stepping == XyceTimeSteppingEnum.REDUCED:
                # try reducing the number of sim steps be setting a fixed step_celling_time
                if self.calculate_initial_conditions:
                    ostream.write('.tran 0 {0} 0 {1}\n\n'.format(time_needed, time_needed/10))
                else:
                    ostream.write('.tran 0 {0} 0 {1} UIC\n\n'.format(time_needed, time_needed/10))
                ostream.write(".options timeint ERROPTION=1 MINTIMESTEPSBP=1 \n\n")

            elif self.params.xyce_parameters.xyce_time_stepping ==XyceTimeSteppingEnum.NOBREAKOPOINTS:
                # add option to ignore breakpoints to run xyce faster, ignoring rise/fall shapes
                if self.calculate_initial_conditions:
                    ostream.write('.tran 0 {0}\n\n'.format(time_needed))
                else:
                    ostream.write('.tran 0 {0} UIC\n\n'.format(time_needed))
                ostream.write(".options timeint ERROPTION=1 BPENABLE=0\n\n")
            else:
                raise ValueError("Unknown timestepping option: "+str(self.params.xyce_parameters.xyce_time_stepping))


            # ************************  set additional simulation options
            # add parallel solver options if using more than 1 core
            if self.params.xyce_parameters.n_processors>1:
                if self.params.xyce_parameters.parallel_solve==True:
                    # ostream.write(".OPTIONS LINSOL TYPE=BELOS TR_GLOBAL_BTF=2 TR_SINGLETON_FILTER=1 TR_AMD=0 TR_PARTITION=0 USE_IFPACK_FACTORY=1\n")
                    ostream.write(".OPTIONS LINSOL TR_GLOBAL_BTF=2 TR_SINGLETON_FILTER=1 TR_AMD=0 TR_PARTITION=0 USE_IFPACK_FACTORY=1\n")
                else:
                    ostream.write(".OPTIONS LINSOL USE_IFPACK_FACTORY=1 TYPE = KLU\n") #TR_SINGLETON_FILTER=0 TR_AMD=0

            else:
                ostream.write(".OPTIONS LINSOL USE_IFPACK_FACTORY=1 TYPE = KLU\n") #TR_SINGLETON_FILTER=0 TR_AMD=0

            # minimize time steps outputted (only enable full output if debugging is on)
            # only output three time steps, t=0, a time to measure the power used, and the final time values
            if not self.params.xyce_parameters.debug_output and not self.params.xyce_parameters.print_all_time_steps:
                # resistor based neurons need all time steps
                if is_memory or (self.params.xyce_parameters.row_neuron.style==NeuronStyleEnum.VCVS_C and is_col_read) or \
                                (self.params.xyce_parameters.col_neuron.style==NeuronStyleEnum.VCVS_C and is_row_read) or is_write:
                    # compute the time at which to measure the power
                    if is_memory:
                        if is_write:
                            intermediate_time = self.params.memory_params.write_eval_time
                        else:
                            intermediate_time = self.params.memory_params.read_eval_time
                    else:
                        intermediate_time = time_needed/2 #TODO: add better times for other cases
                    ostream.write('.OPTIONS OUTPUT INITIAL_INTERVAL = {0} {0} {1}\n'.format(intermediate_time,(time_needed-intermediate_time))) #- self.xyce_params.shared_driver.simulation_settle_time/10

            #add user defined options
            ostream.write(self.params.xyce_parameters.xyce_options+'\n\n')
            ostream.write('.end\n')

        # *************** run xyce simulation
        # start_time = time.time()

        with open(self.screen_file, 'w+b') as screenoutfile:

            #try running xyce in parallel
            run_serial = False
            if (self.params.xyce_parameters.xyce_run_command is None) and (self.params.xyce_parameters.n_processors!=1):
                try:
                    subprocess.check_call(
                                          [self.xyce_params.xyce_parallel_path, '-np', str(self.xyce_params.n_processors), self.xyce_file], cwd=self.dirname,
                                          stdout=screenoutfile, stderr=subprocess.STDOUT,timeout=self.params.xyce_parameters.parallel_timeout
                                          )
                except subprocess.TimeoutExpired:
                    screenoutfile.write("\n***Timeout Expired Killing Sim and Running Serial***\n".encode('utf-8'))
                    run_serial = True # if it fails, run in serial
                    warn("*****Timeout Expired Killing Sim and Running Serial*****")

                except subprocess.CalledProcessError:
                    screenoutfile.write("\n*****Parallel Sim Failed, Running Serial*****\n".encode('utf-8'))
                    warn("*****Parallel Sim Failed, Running Serial*****")
                    run_serial = True # if it fails, run in serial

            try:
                if self.params.xyce_parameters.xyce_run_command is not None:
                    subprocess.check_call(
                                          (self.params.xyce_parameters.xyce_run_command+" "+self.xyce_file), cwd=self.dirname,
                                          stdout=screenoutfile, stderr=subprocess.STDOUT, shell=True
                                          )
                elif self.params.xyce_parameters.n_processors==1 or run_serial==True:
                    subprocess.check_call(
                                          [self.xyce_params.xyce_path, self.xyce_file], cwd=self.dirname,
                                          stdout=screenoutfile, stderr=subprocess.STDOUT
                                          )
            except subprocess.CalledProcessError as cpe:
                cpe.cmd = '{0}: {1}'.format(self.dirname, cpe.cmd)
                screenoutfile.seek(0,0)
                sys.stderr.write(screenoutfile.read().decode('utf-8'))
                raise cpe

        # print("Xyce Run Time is ",time.time()-start_time)

        self.process_xyce_output(self.out_file,self.rows, self.cols, is_row_read, is_col_read, is_write, is_memory)


    def intitialize_xyce_output(self):
        self.rows_output = None
        self.cols_output = None

        self.internal_vars_vs_time = None
        self.xyce_state_vars = None

        self.col_driver_v = None
        self.col_driver_internal_v = None
        self.col_driver_i = None
        self.col_driver_energy = None

        self.row_driver_v = None
        self.row_driver_internal_v = None
        self.row_driver_i = None
        self.row_driver_energy = None

        self.driver_power=0
        self.driver_energy = 0

        self.col_neuron_i = None
        self.row_neuron_i = None

        self.internal_resistances = None
        self.internal_rram_currents =  None
        self.internal_rram_voltages =  None
        self.internal_access_device_voltages =  None
        self.internal_row_voltages =  None

        self.selected_internal_rram_voltages =  None
        self.selected_internal_access_device_voltages =  None


    def process_xyce_output(self, outfile, rows, cols, is_row_read, is_col_read, is_write, is_memory):
        """
        Loads data from a xyce output file.  Simulation options in params must match settings used to run the simulation.
        :param outfile: Xyce output file
        :param rows: numer or rows
        :param cols: number of columns
        :return:
        """
        # **** process output data into internal variables

        # store None to all output variables
        self.intitialize_xyce_output()

        data = XyceData(outfile)

        self.time = data.get_data(1)
        if is_col_read:
            self.rows_output = data.get_data(rows)

        if is_row_read:
            self.cols_output = data.get_data(cols)


        if self.xyce_params.debug_output:
            # save the entire internal variable output
            self.internal_vars_vs_time = data.get_data(rows*cols)
            self.xyce_state_vars = self.internal_vars_vs_time[-1,:].reshape((rows,cols))
            self.internal_vars_vs_time = self.internal_vars_vs_time.reshape((self.time.size,rows,cols))
        else:
            # save only the last time step of the internal variable
            # TODO:  Make xyce output only the final timestep data
            self.xyce_state_vars = data.get_data(rows*cols, -1).reshape((rows,cols))

        if self.xyce_params.debug_output or self.xyce_params.calculate_power:
            # save col driver output if driver exists
            if is_col_read or is_write:
                self.col_driver_v = data.get_data(cols) # voltage at the top node of the crossbar (driver voltage + parasitic)
                self.col_driver_internal_v = data.get_data(cols) # col driver voltage (without parasitic resistance)
                self.col_driver_i = data.get_data(cols) # current through the col driver (positive is current leaving xbar)
                self.col_driver_energy = data.get_data(cols)

            # save row driver output if driver exists
            if is_row_read or is_write:
                self.row_driver_v = data.get_data(rows)  # voltage at the left node of the crossbar (driver voltage + parasitic)
                self.row_driver_internal_v = data.get_data(rows) # row driver voltage (without parastic R)
                self.row_driver_i = data.get_data(rows)    # row driver current (positive is current leaving xbar)
                self.row_driver_energy = data.get_data(rows)

            # calculate and save driver_power
            if self.row_driver_internal_v is not None:
                self.driver_power = -np.sum(self.row_driver_internal_v*self.row_driver_i,axis=1) # use negative sign to acount for current direction
                self.driver_energy = -np.sum(self.row_driver_energy,axis=1)

            if self.col_driver_internal_v is not None:
                self.driver_power -= np.sum(self.col_driver_internal_v*self.col_driver_i,axis=1)
                self.driver_energy -= np.sum(self.col_driver_energy,axis=1)

            #compute/print the write energy for memory writes #TODO:  implement for neural writes/reads
            if is_write and is_memory:
                self.calc_write_energy()


        if self.xyce_params.debug_output:
            # output neuron current if mvm/vmm otherwise output zero
            if is_col_read and self.xyce_params.row_neuron.style==NeuronStyleEnum.VCVS_C and not is_memory:
                self.row_neuron_i = data.get_data(rows)    # current being integrated by the row neuron

            elif is_row_read and self.xyce_params.col_neuron.style==NeuronStyleEnum.VCVS_C and not is_memory:
                self.col_neuron_i = data.get_data(cols)    # current being integrated by the col neuron

            self.internal_resistances = data.get_data(rows*cols).reshape((self.time.size,rows,cols))
            self.internal_rram_currents = data.get_data(rows*cols).reshape((self.time.size,rows,cols))
            # self.internal_rram_voltages = data.get_data(rows*cols).reshape((self.time.size,rows,cols))
            self.internal_access_device_voltages = data.get_data(rows*cols).reshape((self.time.size,rows,cols))
            self.internal_row_voltages = data.get_data(rows*cols).reshape((self.time.size,rows,cols))
            # self.internal_col_voltages = data.get_data(rows*cols).reshape((self.time.size,rows,cols))

        if self.xyce_params.debug_output or self.xyce_params.output_rram_voltages:
            self.internal_rram_voltages = data.get_data(rows*cols).reshape((self.time.size,rows,cols))

        # save individually outputted voltages
        if self.print_locations is not None:
            self.selected_internal_rram_voltages =  data.get_data(len(self.print_locations) )
            self.selected_internal_access_device_voltages = data.get_data(len(self.print_locations) )



    def run_xbar_vmm(self):
        print_debug_calls('XyceCore.run_xbar_vmm')
        with open(self.driver_file, 'w') as ostream:
            self.row_driver.WriteSubckt(ostream, single_pulse=True)
            self.col_driver.WriteSubckt(ostream, single_pulse=True)
            ostream.write('\n\n* row drivers\n')
            self.row_driver.WriteDrivers(ostream, self.__make_row_driver_name(), self.__make_rows_left(), self.vector_vmm, False, resistances=self._driver_resistances(is_row=True, is_read=True))

            # disconnect unused driver rather than High Z
            # ostream.write('\n\n* col drivers\n')
            # self.col_driver.WriteDrivers(ostream, self.__make_col_driver_name(), self.__make_cols_top(), None, False, resistances=self._driver_resistances(is_row=False, is_read=True))
            ostream.write('\n')
        with open(self.neuron_file, 'w') as ostream:
            self.row_neuron.WriteSubckt(ostream)
            self.col_neuron.WriteSubckt(ostream)

            #disconnect unsed neuron rather than using High Z for faster run time
            # ostream.write('\n\n* row neurons\n')
            # self.row_neuron.WriteNeurons(ostream, self.__make_row_neuron_name(), self.__make_rows_right(), self.__make_rows_neuron_output(), True)

            ostream.write('\n\n* col neurons\n')
            self.col_neuron.WriteNeurons(ostream, self.__make_col_neuron_name(), self.__make_cols_bottom(), self.__make_cols_neuron_output(), False)
            ostream.write('\n')
        self.__run(is_row_read=True)

        # convert xyce output to an integrated charge from the neuron and correct for any op-amp issues
        charge = self.col_neuron.IntegrateAndCorrect(self.time, self.cols_output)

        # convert from a total charge to a numeric value
        result = self.row_driver.CorrectOutput(charge, self.xbar.crosspoint.device.Imax)

        # TODO:  Implement this in Xyce rather than after the fact
        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_xbar is True:
            result -= result[0]
        return result
        
    
    def run_xbar_mvm(self):
        # print_debug_calls('XyceCore.run_xbar_mvm')

        with open(self.driver_file, 'w') as ostream:
            self.row_driver.WriteSubckt(ostream, single_pulse=True)
            self.col_driver.WriteSubckt(ostream, single_pulse=True)
            ostream.write('\n\n* row drivers\n')
            #disconnect row driver by seting high Z, values=None
            self.row_driver.WriteDrivers(ostream, self.__make_row_driver_name(), self.__make_rows_left(),values= None, is_write=False, resistances=self._driver_resistances(is_row=True, is_read=True))
            ostream.write('\n\n* col drivers\n')
            # Input the negative vector so that the voltage across the memristor is the same sign as in a vmm
            self.col_driver.WriteDrivers(ostream, self.__make_col_driver_name(), self.__make_cols_top(),values=-self.vector_mvm,is_write=False, resistances=self._driver_resistances(is_row=False, is_read=True))
            ostream.write('\n')
        with open(self.neuron_file, 'w') as ostream:
            self.row_neuron.WriteSubckt(ostream)
            self.col_neuron.WriteSubckt(ostream)
            ostream.write('\n\n* row neurons\n')
            self.row_neuron.WriteNeurons(ostream, self.__make_row_neuron_name(), self.__make_rows_right(), self.__make_rows_neuron_output(), False)

            # #disconnect unused neurons rather than using high Z
            # ostream.write('\n\n* col neurons\n')
            # self.col_neuron.WriteNeurons(ostream, self.__make_col_neuron_name(), self.__make_cols_bottom(), self.__make_cols_neuron_output(), True)

            ostream.write('\n')

        self.__run(is_col_read=True)

        # convert xyce output to an integrated charge from the neuron and correct for any op-amp issues
        charge = self.row_neuron.IntegrateAndCorrect(self.time, self.rows_output)

        # convert from a total charge to a numeric value
        # flip the sign of the output to correct for the fact that we flipped the input vector
        result =-self.col_driver.CorrectOutput(charge, self.xbar.crosspoint.device.Imax)


        # print('  xycecore.run_xbar_mvm returning', result)
        # print('  xycecore.run_xbar_mvm error',np.dot(self._read_matrix(),self.vector_mvm)-result)



        # TODO:  Implement this in Xyce rather than after the fact
        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_xbar is True:
            result -= result[0]
        return result

    
    def update_matrix(self, row_vector, col_vector, learning_rate):
        row_vector,col_vector =  self.clip_and_quantize_update_matrix_inputs(row_vector, col_vector)

        # #####debugging code
        # initial_mat = self._read_matrix()
        # ideal_update = np.outer(row_vector,col_vector)*learning_rate

        # if the learning rate is set, divide it between the row & col according to row_update_portion and tell drivers to scale accordingly
        if learning_rate != 1:
            row_scaling = learning_rate**self.params.algorithm_params.row_update_portion
            col_scaling = learning_rate**(1-self.params.algorithm_params.row_update_portion)
            if self.params.xyce_parameters.row_driver.write_style ==DriverStyleEnum.TIMED:
                #if row driver is timed, scale row update pulse length and increase col vector values
                self.row_driver.duration_scale_factor = row_scaling
                self.col_driver.duration_scale_factor = row_scaling
                col_vector*=col_scaling
            else:
                # if col driver is timed, scale pulse length by col update learning rate and increase row vectors
                self.row_driver.duration_scale_factor = col_scaling
                self.col_driver.duration_scale_factor = col_scaling
                row_vector*=row_scaling
        else:
            self.row_driver.duration_scale_factor=1
            self.col_driver.duration_scale_factor=1

        # normalize all driver values to a 0 to 1 range
        # flip sign of row vector if row driver is level based
        if self.params.xyce_parameters.row_driver.write_style ==DriverStyleEnum.LEVEL:
            row_vector*=-2/self.params.xbar_params.row_update.range
        else:
            row_vector*=2/self.params.xbar_params.row_update.range
        col_vector*=2/self.params.xbar_params.col_update.range




#         print(row_vector)
#         print(col_vector)
#         print(self.internal_vars)
        with open(self.driver_file, 'w') as ostream:
            self.row_driver.WriteSubckt(ostream)
            self.col_driver.WriteSubckt(ostream)
            ostream.write('\n\n* row drivers\n')
            self.row_driver.WriteDrivers(ostream, self.__make_row_driver_name(), self.__make_rows_left(), row_vector, True, resistances=self._driver_resistances(is_row=True))
            ostream.write('\n\n* col drivers\n')
            self.col_driver.WriteDrivers(ostream, self.__make_col_driver_name(), self.__make_cols_top(), col_vector, True, resistances=self._driver_resistances(is_row=False))
            ostream.write('\n')

        # leave blank neuron file to disconnect neurons for write (don't use High Z to reduce sim time)
        with open(self.neuron_file, 'w') as ostream:
            # self.row_neuron.WriteSubckt(ostream)
            # self.col_neuron.WriteSubckt(ostream)
            # ostream.write('\n\n* row neurons\n')
            # self.row_neuron.WriteNeurons(ostream, self.__make_row_neuron_name(), self.__make_rows_right(), self.__make_rows_neuron_output(), True)
            # ostream.write('\n\n* col neurons\n')
            # self.col_neuron.WriteNeurons(ostream, self.__make_col_neuron_name(), self.__make_cols_bottom(), self.__make_cols_neuron_output(), True)
            ostream.write('\n')
        self.__run(is_write=True)
#         print(self.internal_vars)

        # ### debugging code
        # final_mat = self._read_matrix()
        # actual_update = final_mat-initial_mat
        # print("actual update = \n",actual_update)
        # print("ideal update = \n",ideal_update)
        # print(" update error = " ,(actual_update-ideal_update))
        # print("init_mat = \n",initial_mat)



    def _read_matrix(self):
        print_debug_calls('XyceCore._read_matrix',self.xyce_state_vars)
#         print('\n\ninternal:\n',self.internal_vars)
#         print('corrected:\n',self.xbar.crosspoint.device.UncalculateXyceStateVar(self.internal_vars))
        if self.xyce_state_vars is None:
            raise ValueError("The matrix has not yet been set, cannot read")
        return self.xbar.crosspoint.device.UncalculateXyceStateVar(self.xyce_state_vars)
    
    
    def _save_matrix(self):
        return self.xyce_state_vars.copy()
    

    def _restore_matrix(self, matrix):
        self.xyce_state_vars = matrix.copy()

# ************************** xyce analysis functions
    def calc_write_energy(self):
        """
        Computes the write energy and driver_power
        :return:
        """
        self.write_energy = self.driver_energy[-1]

        self.write_energy_debug = np.trapz(self.driver_power,self.time[:,0])

        #calculate the time at which to take the driver_power
        write_time = self.params.memory_params.write_eval_time
        #find the time index closest to the desired write time
        idx = (np.abs(self.time-write_time)).argmin()
        self.write_power = self.driver_power[idx]
        return self.write_energy, self.write_power

    def calc_rise_and_fall_time(self, voltages, is_write, is_memory,threshold = 0.99):
        """
        Calculate the rise and fall times based on the voltages passed

        :param Voltages:
        :param is_write:
        :param is_memory:
        :param threshold:  What fraction of max voltage should be achieved.
        :return:
        """

        if is_memory:
            #time at start of pulse
            start_time =self.xyce_params.shared_driver.initial_wait_time
            if is_write:
                Vmax_time = self.params.memory_params.write_eval_time
                idx = (np.abs(self.time-Vmax_time)).argmin()
                Vmax = voltages[idx]
                Vmin = voltages[0]
                #time at end of pulse
                end_time =self.xyce_params.shared_driver.initial_wait_time+self.xyce_params.shared_driver.rise_time+self.params.memory_params.write_time
            else:
                raise NotImplementedError("this is implemented but not tested")
                Vmax_time = self.params.memory_params.read_eval_time
                idx = (np.abs(self.time-Vmax_time)).argmin()
                Vmax = voltages[idx]
                Vmin = 0
                end_time =self.xyce_params.shared_driver.initial_wait_time+self.xyce_params.shared_driver.rise_time+self.params.memory_params.read_time
            #time at end of simulation or before next pulse starts
            stop_time = end_time +self.xyce_params.shared_driver.fall_time+self.xyce_params.shared_driver.simulation_settle_time
        else:
            raise NotImplementedError

        # find indicies of relevant times to define search ranges
        idx_start = (np.abs(self.time-start_time)).argmin()
        idx_end = (np.abs(self.time-end_time)).argmin()
        idx_stop = (np.abs(self.time-stop_time)).argmin()

        # find rise time
        Vthreshold = (Vmax-Vmin)*threshold+Vmin

        aa=voltages[idx_start:idx_end]
        bb=self.time[idx_start:idx_end]
        func = interp1d(np.squeeze(voltages[idx_start:idx_end]),np.squeeze(self.time[idx_start:idx_end]),kind='linear')
        rise_time  = func(Vthreshold)-start_time


        # idx_rise = (np.abs(voltages[idx_start:idx_end]-Vthreshold) ).argmin()+idx_start
        # rise_time  = self.time[idx_rise]-start_time

        # find the fall time
        Vthreshold = (Vmax-Vmin)*(1-threshold)+Vmin

        x = np.squeeze(voltages[idx_stop:idx_end:-1])
        y= np.squeeze(self.time[idx_stop:idx_end:-1])
        func = interp1d(x,y,kind='linear')
        fall_time  = func(Vthreshold)-end_time

        # idx_fall = (np.abs(voltages[idx_end:idx_stop]-Vthreshold) ).argmin()+idx_end
        # fall_time  = self.time[idx_fall]-end_time

        return rise_time, fall_time

# *************************** memory core functions

    def memory_write(self, row_voltages, col_voltages, precharge_row=0, precharge_col=0):
        """
        writes a memory crossbar using specified voltages, drive from left and top
        :param row_voltages: voltages to apply to the rows
        :param col_voltages: voltages to apply to the columns
        :param precharge_row: The initial / final voltage on the row.  The line starts/ends at this voltage (used for calculating write times)
        :param precharge_col: The initial / final voltage on the col.  The line starts/ends at this voltage
        :return:
        """

        # calculate initial conditions if precharging line:
        if precharge_row !=0 or precharge_col!=0:
            self.calculate_initial_conditions = True

        # write the driver files
        with open(self.driver_file, 'w') as ostream:
            self.row_driver.WriteSubckt(ostream, single_pulse=True, is_memory=True)
            self.col_driver.WriteSubckt(ostream, single_pulse=True, is_memory=True)
            ostream.write('\n\n* row drivers\n')
            self.row_driver.WriteDrivers(ostream, self.__make_row_driver_name(), self.__make_rows_left(), row_voltages, is_write=True, is_memory=True, resistances=self._driver_resistances(is_row=True),Vprecharge=precharge_row)
            ostream.write('\n\n* col drivers\n')
            self.col_driver.WriteDrivers(ostream, self.__make_col_driver_name(), self.__make_cols_top(), col_voltages, is_write=True, is_memory=True, resistances=self._driver_resistances(is_row=False),Vprecharge=precharge_col)
            ostream.write('\n')

        # write blank file for the neurons
        with open(self.neuron_file, 'w') as ostream:
            ostream.write('\n')

        self.__run(is_write=True, is_memory=True)

        # turn off initial condition calculation
        self.calculate_initial_conditions = False


    def memory_read(self, row_voltages, col_voltages, row_impedances,col_impedances, read_row):
        """
        reads a memory crossbar using specified voltages
        :param row_voltages: voltages to apply to the rows on left, read bottom
        :param col_voltages: voltages to apply to the columns on top, read right
        :return: the output voltage on the read resistor
        """
        # write the driver files
        with open(self.driver_file, 'w') as ostream:
            self.row_driver.WriteSubckt(ostream, single_pulse=True, is_memory=True)
            self.col_driver.WriteSubckt(ostream, single_pulse=True, is_memory=True)

            #write row or column drivers
            if read_row:
                ostream.write('\n\n* row drivers\n')
                self.row_driver.WriteDrivers(ostream, self.__make_row_driver_name(), self.__make_rows_left(), row_voltages, is_write=False, is_memory=True, resistances=self._driver_resistances(is_row=True, is_read=True))
            else:
                ostream.write('\n\n* col drivers\n')
                self.col_driver.WriteDrivers(ostream, self.__make_col_driver_name(), self.__make_cols_top(), col_voltages, is_write=False, is_memory=True, resistances=self._driver_resistances(is_row=False, is_read=True))
            ostream.write('\n')

        # write the neurons
        with open(self.neuron_file, 'w') as ostream:
            self.row_neuron.WriteMemSubckt(ostream)
            self.col_neuron.WriteMemSubckt(ostream)

            if read_row:
                ostream.write('\n\n* col neurons\n')
                self.col_neuron.WriteMemNeurons(ostream, self.__make_col_neuron_name(), self.__make_cols_bottom(), self.__make_cols_neuron_output(), col_voltages,col_impedances, series_resistances=self._driver_resistances(is_row=False, is_output=True, is_read=True)) # add driver resistance for parasitic compensation
            else:
                ostream.write('\n\n* row neurons\n')
                self.row_neuron.WriteMemNeurons(ostream, self.__make_row_neuron_name(), self.__make_rows_right(), self.__make_rows_neuron_output(), row_voltages, row_impedances, series_resistances=self._driver_resistances(is_row=True, is_output=True, is_read=True))
            ostream.write('\n')

        # find the output voltage
        if read_row:
            self.__run(is_write=False, is_memory=True, is_row_read=True)
            result = self.col_neuron.CalcMemOutput(self.time,self.cols_output)-col_voltages  # subtract supply voltage to get voltage across the read resistor
        else:
            self.__run(is_write=False, is_memory=True, is_col_read=True)
            result = self.row_neuron.CalcMemOutput(self.time,self.rows_output)-row_voltages # subtract supply voltage to get voltage across the read resistor

        return result