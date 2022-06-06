#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from copy import copy
from warnings import warn

import numpy as np
from scipy.optimize import brentq


class MemoryCore(object):
    '''
    built assuming a xyce core
    '''

    def __init__(self, clipper_core_factory, params):
        """

        :param clipper_core_factory:
        :param params:
        :type params: Parameters
        :return:
        """

        self.core = clipper_core_factory()
        """:type: XyceCore"""

        self.params = params

        self.clipper_core_factory = clipper_core_factory
        # save reference to core for shorter access
        self.memory_params = params.memory_params


    def set_matrix(self, matrix):
        matrix = np.array(matrix, dtype="float32")  # create a numpy copy (force float32 to avoid ints coming through)
        return self.core.set_matrix(matrix)

    def read(self, active_rows, active_cols, read_row=True):
        """
        Reads the memory
        :param active_rows: index of active row
        :param active_cols: indices of active cols
        :param: read_row: apply read voltage to row and readout along the column
        :return: the output voltage on the read resistor
        """
        Vrow = self.memory_params.Vrow_read
        Vcol = self.memory_params.Vcol_read
        Vword = self.memory_params.Vword_read
        Vbit = self.memory_params.Vbit_read
        drive_impedance = self.memory_params.drive_impedance
        read_impedance = self.memory_params.read_impedance
        highz_impedance = self.memory_params.highz_impedance





        if Vrow is None:
            row_voltages = np.zeros(self.core.rows)
            row_impedances = np.ones(self.core.cols)*highz_impedance
        else:
            row_voltages = np.ones(self.core.rows)*Vrow
            row_impedances = np.ones(self.core.cols)*0  # series drive impedance added in every driver


        row_voltages[[active_rows]]=Vword

        if Vcol is None:
            col_voltages = np.zeros(self.core.cols)
            col_impedances = np.ones(self.core.cols)*highz_impedance
        else:
            col_voltages = np.ones(self.core.cols)*Vcol
            col_impedances = np.ones(self.core.cols)*0

        col_voltages[[active_cols]]=Vbit

        # set the readout resistance on rows/cols being readout
        if read_row==True:
            col_impedances[[active_cols]]=read_impedance
        else:
            row_impedances[[active_rows]]=read_impedance


        result = self.core.memory_read(row_voltages,col_voltages, row_impedances,col_impedances, read_row)

        # return only the active rows/cols being read
        if read_row:
            return result[active_cols]
        else:
            return result[active_rows]


    def write(self, active_rows, active_cols, flip_polarity=False, precharge_inputs=False):
        """
        Writes a memory based on specified voltages (apply Vrow, col, word and bit through driver resistance)
        :param active_rows: index of active rows
        :param active_cols: indices of active cols
        :param flip_polarity: flip row and col voltages so that the opposite sign write is done
        :param precharge_inputs:  If true, the rows/cols are precharged, they all start at the unselect voltage and only
                                  the selected device is switched and is then returned to the precharge voltage
        :return:
        """

        if flip_polarity:
            Vrow = self.memory_params.Vcol_write
            Vcol = self.memory_params.Vrow_write
            Vword = self.memory_params.Vbit_write
            Vbit = self.memory_params.Vword_write
        else:
            Vrow = self.memory_params.Vrow_write
            Vcol = self.memory_params.Vcol_write
            Vword = self.memory_params.Vword_write
            Vbit = self.memory_params.Vbit_write

        row_voltages = np.ones(self.core.rows)*Vrow
        row_voltages[[active_rows]]=Vword

        col_voltages = np.ones(self.core.cols)*Vcol
        col_voltages[[active_cols]]=Vbit

        if precharge_inputs:
            self.core.memory_write(row_voltages,col_voltages,precharge_row=Vrow, precharge_col=Vcol)
        else:
            self.core.memory_write(row_voltages,col_voltages)



    def set_mem_params(self, Ileak = None, write_time = None, read_time = None,
            Vword_read=None, Vbit_read=None, Vrow_read=None, Vcol_read=None,
            Vword_write=None, Vbit_write=None, Vrow_write=None, Vcol_write=None,
            drive_impedance=None, read_impedance=None, highz_impedance=None):


        if Ileak is not None:
            self.memory_params.Ileak=Ileak

        if write_time is not None:
            self.memory_params.write_time=write_time
        if read_time is not None:
            self.memory_params.read_time=read_time

        if Vword_read is not None:
            self.memory_params.Vword_read=Vword_read
        if Vbit_read is not None:
            self.memory_params.Vbit_read=Vbit_read
        if Vrow_read is not None:
            self.memory_params.Vrow_read=Vrow_read
        if Vcol_read is not None:
            self.memory_params.Vcol_read=Vcol_read

        if Vword_write is not None:
            self.memory_params.Vword_write=Vword_write
        if Vbit_write is not None:
            self.memory_params.Vbit_write=Vbit_write
        if Vrow_write is not None:
            self.memory_params.Vrow_write=Vrow_write
        if Vcol_write is not None:
            self.memory_params.Vcol_write=Vcol_write

        if drive_impedance is not None:
            self.memory_params.drive_impedance=drive_impedance
        if read_impedance is not None:
            self.memory_params.read_impedance=read_impedance
        if highz_impedance is not None:
            self.memory_params.highz_impedance=highz_impedance


    def calculate_unselect_voltage(self,Ileak, nrows,ncols):
        """
        Return the voltage required to get Ileak current across (nrows)*(ncols)devices
        :param Ileak:
        :param nrows:
        :param ncols:
        :return:
        """
        return self.core.xbar.crosspoint.device.crosspoint_voltage_given_current(Ileak/((nrows)*(ncols)),self.core.xbar.crosspoint.device.x_max)


    def find_write_voltages(self, Ileak, Vwrite_device, nrows=None, ncols=None, row=None, col=None):
        """

        :return:
        """

        if nrows is None:
            nrows=self.core.rows
        if ncols is None:
            ncols=self.core.cols

        # save state before write
        old_states =self.core._save_matrix() # creates a copy of the matrix

        #set matrix to all 1's
        weights = np.ones([nrows,ncols])
        self.set_matrix(weights)

        Vunselect = self.calculate_unselect_voltage(Ileak,nrows,ncols)
        print("The unselected voltage is "+str(Vunselect))

        V_ideal = self.core.xbar.crosspoint.device.crosspoint_voltage_given_vrram(Vwrite_device,
                                                                        self.core.xbar.crosspoint.device.x_max)
        V_guess =(self.params.xyce_parameters.xbar.wire_resistivity*self.params.xyce_parameters.xbar.cell_spacing) *\
                 self.core.xbar.crosspoint.device.rram_current(Vwrite_device,'max')*(nrows+ncols)+V_ideal

        print("The ideal write voltage without parasitics is "+str(V_ideal))
        print("The estimated write voltage with parasitics is "+str(V_guess))


        # find the write voltage through simulations
        if row is None:
            row = nrows-1
        if col is None:
            col = ncols-1

        func  = lambda Vwrite: self.find_VRRAM_write(Vunselect,Vwrite,row,col)-Vwrite_device
        # Vwrite = newton(func, x0=V_guess, tol=1e-3,maxiter=10)
        Vwrite = brentq(func, a=V_ideal, b= 5*V_guess, rtol=1e-4,maxiter=10)

        print("The write voltage is "+str(Vwrite))

        # restore state after write test
        self.core._restore_matrix(old_states)

        return Vunselect, Vwrite


    def find_VRRAM_write(self, Vunselect, Vwrite, row, col, return_V_AD=False):
        """
        Find the voltage across the indicated device, runs a xyce simulation, uses currently stored matrix
        Assumes device is in LRS (applies a set pulse)

        :param Vunselect:
        :param Vwrite:
        :return:
        """
        print("debug: Vwrite Test Iteration= "+str(Vwrite))

        # output voltages at desired row, col
        self.core.print_locations = [(row, col)]

        # save state before write
        old_states =self.core._save_matrix()

        # set write parameters:
        self.set_mem_params(Vword_write=Vwrite,Vbit_write=0,Vrow_write=Vwrite/2-Vunselect/2,Vcol_write=Vwrite/2+Vunselect/2)
        # do write
        self.write([row],[col],flip_polarity=False)

        #find the time index closest to the desired write time
        time = self.core.time
        idx = (np.abs(time-self.memory_params.write_eval_time)).argmin()

        #find the voltage across the RRAM and Access Device
        Vrram = self.core.selected_internal_rram_voltages[idx]
        V_AD = self.core.selected_internal_access_device_voltages[idx]


        #verify voltage was stable to within 1%
        Vrram2 = self.core.selected_internal_rram_voltages[idx+1]
        if self.params.xyce_parameters.print_all_time_steps == True:  #can't perform check if time steps are not printed out
            if np.abs(Vrram2-Vrram)/Vrram>0.01:
                warn("The write time was too short or state change too large and so the voltage is still changing")

        # restore state after write test
        self.core._restore_matrix(old_states)
        print("debug: Vrram= "+str(Vrram))

        # disable location printing
        self.core.print_locations = None

        if return_V_AD:
            return Vrram, V_AD
        else:
            return Vrram




    def find_read_voltage(self, Vread_device, Rload, nrows=None, ncols=None):
        """
        Finds worst case read disturb voltage to set the maximum read voltage

        :return:
        """

        if nrows is None:
            nrows=self.core.rows
        if ncols is None:
            ncols=self.core.cols

        row = nrows-1
        col = 0

        # save state before read
        old_states =self.core._save_matrix() # creates a copy of the matrix

        # initialize matrix to worst case read disturb
        weights = np.zeros([nrows,ncols])
        self.set_matrix(weights)

        # find voltage across ReRAM and Access Device
        V_xpoint = self.core.xbar.crosspoint.device.crosspoint_voltage_given_vrram(Vread_device,
                                                                        self.core.xbar.crosspoint.device.x_min)

        V_guess =(Rload) *self.core.xbar.crosspoint.device.rram_current(Vread_device,'min')+V_xpoint



        print("The ideal voltage at the crosspoint is "+str(V_xpoint))
        print("The estimated read voltage with the load is "+str(V_guess))


        # find the write voltage through simulations

        func  = lambda Vread: self.find_VRRAM_read(row, col, Vread, Rload)[0] -Vread_device
        # Vwrite = newton(func, x0=V_guess, tol=1e-3,maxiter=10)
        Vread = brentq(func, a=V_guess, b= 2*V_guess, rtol=5e-5,maxiter=10)

        print("The read voltage is "+str(Vread))

        # restore state after write test
        self.core._restore_matrix(old_states)

        return Vread

    def find_read_margins(self, rows, cols):
        """
        Finds the read margins voltages across the ReRAM for the closest and furthest devices.  Matrix programmed to give the worst case read margin for each case

        :param rows:
        :param cols:
        :return: returns voltages across the read resistors and then the voltages across the ReRAMs themselves
        """


                # ***** find worst case read margin
        # HRS
        weights = np.zeros([rows,cols])
        #center bottom/left
        weights[1:,:-1]=1
        # top row
        weights[0,:]=0
        #right col
        weights[:,-1]=1
        # top right entry
        weights[0,-1]=0
        self.set_matrix(weights)
        V_HRS_furthest_RRAM, V_HRS_furthest = self.find_VRRAM_read(row=0,col=cols-1,read_row=True)
        print("V_HRS_furthest_RRAM= "+str(V_HRS_furthest_RRAM)+" , V_HRS_furthest= "+str(V_HRS_furthest))

        # LRS
        weights = np.zeros([rows,cols])
        # center bottom/left
        weights[1:,:-1]=0
        # top row
        weights[0,:]=1
        # right col
        weights[:,-1]=0
        # top right entry
        weights[0,-1]=1
        self.set_matrix(weights)
        V_LRS_furthest_RRAM, V_LRS_furthest = self.find_VRRAM_read(row=0,col=cols-1,read_row=True)
        print("V_LRS_furthest_RRAM= "+str(V_LRS_furthest_RRAM)+" , V_LRS_furthest= "+str(V_LRS_furthest))


        # ***** find closest case read margin
        # HRS
        weights = np.zeros([rows,cols])
        #center top/right
        weights[:-1,1:]=1
        # bottom row
        weights[-1,:]=0
        #left col
        weights[:,0]=1
        # bottom left entry
        weights[-1,0]=0
        self.set_matrix(weights)
        V_HRS_closest_RRAM, V_HRS_closest = self.find_VRRAM_read(row=rows-1,col=0,read_row=True)
        print("V_HRS_closest_RRAM= "+str(V_HRS_closest_RRAM)+" , V_HRS_closest= "+str(V_HRS_closest))

        # LRS
        weights = np.zeros([rows,cols])
        #center top/right
        weights[:-1,1:]=0
        # bottom row
        weights[-1,:]=1
        #left col
        weights[:,0]=0
        # bottom left entry
        weights[-1,0]=1

        self.set_matrix(weights)
        V_LRS_closest_RRAM, V_LRS_closest = self.find_VRRAM_read(row=rows-1,col=0,read_row=True)
        print("V_LRS_closest_RRAM= "+str(V_LRS_closest_RRAM)+" , V_LRS_closest= "+str(V_LRS_closest))

        return V_HRS_furthest, V_LRS_furthest, V_HRS_closest, V_LRS_closest, \
               V_HRS_furthest_RRAM, V_LRS_furthest_RRAM, V_HRS_closest_RRAM, V_LRS_closest_RRAM


    def find_VRRAM_read(self, row, col, Vread=None, Rload=None, read_row=True):
        """
        Find the voltage across the indicated device, runs a xyce simulation, uses currently stored matrix
        (restores matrix if there is a read disturb)

        :param Vunselect:
        :param Vwrite:
        :return: (Vrram, Vout)  Vrram = voltage across the RRAM itself  Vout = voltage across the read resistor
        """
        if Vread is not None: print("debug: Vread Test Iteration= "+str(Vread))

        # output voltages at desired row, col
        self.core.print_locations = [(row, col)]

        # save state before write
        old_states =self.core._save_matrix()


        # do a memory read

        # set read parameters, if specified:
        if Vread is not None:
            self.set_mem_params(Vword_read=Vread/2,Vbit_read=-Vread/2,Vrow_read=0,Vcol_read=0)
        if Rload is not None:
            self.set_mem_params(Vword_read=Vread/2,Vbit_read=-Vread/2,Vrow_read=0,Vcol_read=0, read_impedance=Rload)


        # do read
        Vout = self.read([row],[col], read_row=read_row)


        # find the time index closest to the desired read time
        time = self.core.time
        idx = (np.abs(time-self.memory_params.read_eval_time)).argmin()

        # find the voltage across the RRAM
        Vrram = self.core.selected_internal_rram_voltages[idx]

        # verify voltage was stable to within 1%
        Vrram2 = self.core.selected_internal_rram_voltages[idx+1]
        if self.params.xyce_parameters.print_all_time_steps == True:  #can't perform check if time steps are not printed out
            if np.abs(Vrram2-Vrram)/Vrram>0.01:
                warn("The read time was too short or state change too large and so the voltage is still changing")

        # restore state after read test
        self.core._restore_matrix(old_states)
        if Vread is not None: print("debug: Vrram= "+str(Vrram))

        # disable location printing
        self.core.print_locations = None

        return Vrram, Vout


    def _read_matrix(self):
        output = self.core._read_matrix()
        return output.copy()


    def _save_matrix(self):
        output = self.core._save_matrix()
        return output.copy()

    def _restore_matrix(self, matrix):
        return self.core._restore_matrix(matrix)
