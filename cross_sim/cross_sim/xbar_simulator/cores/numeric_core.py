#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

from .clipper_core import ClipperCore
from .debug import print_debug_calls, DEBUG_CALLS
from ..parameters.parameter_defaults import WriteNoiseModelEnum, UpdateModelEnum
from warnings import warn
import time

DEBUG_NUMERIC = False
DEBUG_NUMERIC = DEBUG_NUMERIC and DEBUG_CALLS

class NumericCore(ClipperCore):
    '''
    An inner :py:class:`.ICore` that performs purely-numeric calculations
    '''

    def __init__(self, params):
        self.matrix = None
        self.mcopy = None  # scratch space to store matrix during computations without repeatedly re-allocating space.
        self.vector_vmm = None
        self.vector_mvm = None
        self.matrix_temp = None
        self.par_mask = None
        self.matrix_error = None

        ClipperCore.__init__(self, params)

        global ncp
        if self.params.numeric_params.useGPU:
            global cp
            import cupy as cp
            cp.cuda.Device(self.params.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np


    def set_matrix(self, matrix, applyErrors=True):

        matrix = self.clip_matrix(matrix)

        if self.params.numeric_params.useGPU:
            self.matrix = cp.array(matrix)
        else:
            self.matrix = matrix

        # If doing training and SIMD parasitics, create a mask here
        if self.params.numeric_params.Nex_par > 1:
            Nex_par = self.params.numeric_params.Nex_par
            Nx,Ny = matrix.shape
            self.par_mask = ncp.zeros((Nex_par*Nx,Nex_par*Ny),dtype=self.matrix.dtype)
            for m in range(Nex_par):
                x_start, x_end = m*Nx, (m+1)*Nx
                y_start, y_end = m*Ny, (m+1)*Ny
                self.par_mask[x_start:x_end,y_start:y_end] = 1
            self.mask_nnz = ncp.count_nonzero(self.par_mask)
            self.par_mask = (self.par_mask > 1e-9)

        # Apply weight error
        if applyErrors:
            self._apply_weight_errors()


    def set_vmm_inputs(self, vector):
        # clip and quantize vector
        vector = self.clip_vmm_inputs(vector)
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        # clip and quantize vector
        # for convolution, this is done in batch for all sliding windows inside ConvolutionCore for faster processing
        if not self.params.convolution_parameters.is_conv_core:
            vector = self.clip_mvm_inputs(vector)
        self.vector_mvm = vector

    
    def run_xbar_vmm(self,vector=None):
        print_debug_calls('NumericCore.run_xbar_vmm')
        if DEBUG_NUMERIC:
            print(self.vector_vmm)
            print('{dot}')
            print(self.matrix)
            print('=')
            print(self.vector_vmm.dot(self.matrix))

        # apply read noise
        noisy_matrix = self.read_noise_matrix()

        # Calculate VMM using dot product
        # Parasitics
        if self.params.numeric_params.Rp > 0 and self.params.numeric_params.parasitic_backprop:
            solved = False
            while not solved:
                solved = True
                try:
                    result = self.xbar_mvm_parasitics(self.vector_vmm,noisy_matrix.transpose())
                except ValueError:
                    self.params.numeric_params.convergence_param *= 0.98
                    if self.params.numeric_params.convergence_param <= 1e-2:
                        print("VMM failed to converge")
                    print("Reduced MVM convergence param to: "+str(self.params.numeric_params.convergence_param))
                    solved = False

        else:
            if self.params.numeric_params.useGPU:
                result = cp.dot(self.vector_vmm,noisy_matrix)
            elif self.params.numeric_params.useEINSUM:
                result = np.einsum('i,ij->j',self.vector_vmm,noisy_matrix)
            else:
                result = self.vector_vmm.dot(noisy_matrix)

        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_offset_xbar is True:
            result -= result[0]

        return result


    def run_xbar_mvm(self,vector=None):
        # print_debug_calls('NumericCore.run_xbar_mvm')
        if DEBUG_NUMERIC:
            print(self.matrix)
            print('{dot}')
            print(self.vector_mvm)
            print('=')
            print(self.matrix.dot(self.vector_mvm))

        # Apply read noise (unique noise on each call)
        noisy_matrix = self.read_noise_matrix()

        # Load input vector
        if vector is None:
            vector = self.vector_mvm

        # Call pseudo-circuit simulator if parasitic resistance is enabled
        if self.params.numeric_params.Rp > 0 and vector.any():
            solved, retry = False, False
            while not solved:
                solved = True
                try:
                    result = self.xbar_mvm_parasitics(vector,noisy_matrix.copy())
                except RuntimeError:
                    solved, retry = False, True
                    self.params.numeric_params.convergence_param *= 0.98
                    if self.params.numeric_params.convergence_param <= 1e-2:
                        raise ValueError("Parasitic MVM failed to converge")
            if retry:
                print("Reduced MVM convergence parameter to: "+str(self.params.numeric_params.convergence_param))

        else:
            # Compute using matrix vector dot product
            if self.params.numeric_params.useGPU:
                result = cp.dot(noisy_matrix,vector)
            elif self.params.numeric_params.useEINSUM:
                result = np.einsum('i,ij->j',vector,noisy_matrix.T)
            else:
                result = noisy_matrix.dot(vector)
        
        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_offset_xbar:
            result -= result[0]

        return result


    """
    This function is used for calculating the MVM where the positive and negative weights are interleaved
    across the columns of one array rather than existing in separate arrays. This is used to reduce parasitic
    voltage drops.
    By convention, this function will be called on the positive core. The negative core will be passed in
    as an argument.
    """
    def run_xbar_mvm_interleaved(self,core_neg,vector=None):

        # Apply read noise to the positive core
        matrix_pos = self.read_noise_matrix()
        matrix_neg = core_neg.read_noise_matrix()

        if vector is None:
            vector = self.vector_mvm

        # Calculate MVM using dot product
        if self.params.numeric_params.Rp > 0 and vector.any():
            solved, retry = False, False
            while not solved:
                solved = True
                try:
                    result = self.xbar_mvm_parasitics_columnOnly_interleaved(vector,matrix_pos.copy(),matrix_neg.copy())
                except RuntimeError:
                    solved, retry = False, True
                    self.params.numeric_params.convergence_param *= 0.98
                    if self.params.numeric_params.convergence_param <= 1e-2:
                        raise ValueError("Parasitic MVM failed to converge")
            if retry:
                print("Reduced MVM convergence parameter to: "+str(self.params.numeric_params.convergence_param))

        else:
            if self.params.numeric_params.useGPU:
                result = cp.dot(matrix_pos-matrix_neg,vector)
            else:
                matrix_posneg = matrix_pos - matrix_neg
                if self.params.numeric_params.useEINSUM:
                    result = np.einsum('i,ij->j',vector,matrix_posneg.T)
                else:
                    result = matrix_posneg.dot(vector)

        return result


    def xbar_mvm_parasitics(self,vector,matrix):
        """
        Calculates the MVM result including parasitic resistance

        vector : input vector
        matrix : weight matrix (input the transpose for VMM)
        """
        if self.params.numeric_params.circuit.noRowParasitics and self.params.numeric_params.circuit.Vselect == 0:
            return self.xbar_mvm_parasitics_columnOnly(vector,matrix)

        # Parasitic resistance
        # Expressed in ohms; if weight range is [0.1,1], then the resistance of the memory elements is in the range [1,10]
        # A practical maximum value of Rp is Rp = 1/(Nx*Ny)
        Rp = self.params.numeric_params.Rp
        Niters_max = self.params.numeric_params.Niters_max_parasitics
        Verr_th = self.params.numeric_params.Verr_th_mvm
        gamma = self.params.numeric_params.convergence_param
        noRowParasitics = self.params.numeric_params.circuit.noRowParasitics
        Vselect = self.params.numeric_params.circuit.Vselect
        Vread = self.params.numeric_params.circuit.Vread
        
        # If SIMD, retrieve the appropriate masking matrix
        useMask = (self.params.numeric_params.Nex_par > 1 and self.matrix_temp is not None)
        if useMask:
            mask = self.par_mask

        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Initial estimate of device currents
        # Input seen at every element
        dV0 = ncp.tile(vector,(matrix.shape[0],1))
        Ires = matrix*dV0
        dV = dV0.copy()

        # Iteratively calculate parasitics and update device currents
        while Verr > Verr_th and Niters < Niters_max:
            # Calculate parasitic voltage drops
            if useMask:
                Isum_col = mask*ncp.cumsum(Ires,1)
                Isum_row = mask*ncp.cumsum(Ires[::-1],0)[::-1]
            else:
                Isum_col = ncp.cumsum(Ires,1)
                Isum_row = ncp.cumsum(Ires[::-1],0)[::-1]

            Vdrops_col = Rp*ncp.cumsum(Isum_col[:,::-1],1)[:,::-1]
            Vdrops_row = Rp*ncp.cumsum(Isum_row,0)
            Vpar = Vdrops_col + Vdrops_row

            # Calculate the error for the current estimate of memristor currents
            VerrMat = dV0 - Vpar - dV

            # Evaluate overall error; if using SIMD, make sure only to count the cells that matter
            if useMask:
                Verr = ncp.max(ncp.abs(VerrMat[mask]))
            else:
                Verr = ncp.max(ncp.abs(VerrMat))
            if Verr < Verr_th:
                break

            # Update memristor currents for the next iteration
            dV += gamma*VerrMat
            if Vselect > 0:
                dV = (dV - Vselect)*(dV > Vselect) + (dV + Vselect)*(dV < -Vselect) 
            Ires = matrix*dV
            Niters += 1
            
        # Calculate the summed currents on the columns
        Icols = ncp.sum(Ires,axis=1)

        # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
        if Verr > Verr_th:
            raise RuntimeError('Parasitic resistance too high: could not converge!')
        del Ires
        return Icols


    def xbar_mvm_parasitics_columnOnly(self,vector,matrix):
        """
        Calculates the MVM result including parasitic resistance
        Assumes an array topology where input is applied bitwise on the gate, and there is no parasitic resistance
        along a column (noRowParasitics = True, Vselect = 0)

        vector : input vector
        matrix : weight matrix (input the transpose for VMM)
        """

        # Parasitic resistance
        Rp = self.params.numeric_params.Rp
        Niters_max = self.params.numeric_params.Niters_max_parasitics
        Verr_th = self.params.numeric_params.Verr_th_mvm
        gamma = self.params.numeric_params.convergence_param
            
        # If SIMD, retrieve the appropriate masking matrix
        useMask = (self.params.numeric_params.Nex_par > 1 and self.matrix_temp is not None)
        if useMask:
            mask = self.par_mask

        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Initial estimate of device currents
        dV0 = ncp.tile(vector,(matrix.shape[0],1))

        matrix0 = matrix.copy()

        # FOR A THREE-TERMINAL DEVICE:
        # If input is zero, the device is gated off and cannot conduct current regardless of the drain voltage
        # Ensure this by zeroing out the conductances where the row is not activated
        active_inputs = (ncp.abs(dV0) > 1e-9)
        matrix *= active_inputs
        Ires = matrix*dV0
        dV = dV0.copy()

        # Iteratively calculate parasitics and update device currents
        while Verr > Verr_th and Niters < Niters_max:

            # # Calculate parasitic voltage drops
            Isum_col = ncp.cumsum(Ires,1)
            if useMask:
                Isum_col *= mask

            Vpar = ncp.flip(Isum_col, 1)
            ncp.cumsum(Vpar, 1, out=Vpar)
            Vpar = ncp.flip(Vpar, 1)
            Vpar *= Rp

            # Calculate the error for the current estimate of memristor currents
            # If using SIMD, make sure only to count the cells that matter
            VerrMat = dV0 - Vpar - dV
            if useMask:
                Verr = ncp.max(ncp.abs(VerrMat[mask]))
            else:
                Verr = ncp.max(ncp.abs(VerrMat))

            if Verr < Verr_th:
                break

            # Update memristor currents for the next iteration
            # Under-relaxation
            dV += gamma*VerrMat

            # Over-relaxation
            # dV = (1-gamma)*dV + gamma*(dV0 - Vpar)

            Ires = matrix*dV
            Niters += 1

        # Calculate the summed currents on the columns
        Icols = ncp.sum(Ires,axis=1)

        if Verr > Verr_th:
            raise RuntimeError('Parasitic resistance too high: could not converge!')
        del Ires
        return Icols


    def xbar_mvm_parasitics_columnOnly_interleaved(self,vector,matrix_pos,matrix_neg):
        """
        Calculates the MVM result including parasitic resistance on the bit line

        vector : input vector
        matrix_pos : positive weight matrix
        matrix_neg : negative weight matrix
        """
        # For efficiency, this code assumes no select device and no row parasitics
        if not self.params.numeric_params.circuit.noRowParasitics or self.params.numeric_params.circuit.Vselect > 0:
            raise ValueError("Interleaved parasitics option requires no row parasitics and Vselect = 0")

        # Parasitic resistance
        Rp = self.params.numeric_params.Rp
        Niters_max = self.params.numeric_params.Niters_max_parasitics
        Verr_th = self.params.numeric_params.Verr_th_mvm
        gamma = self.params.numeric_params.convergence_param
        
        # If SIMD, retrieve the appropriate masking matrix
        useMask = (self.params.numeric_params.Nex_par > 1 and self.matrix_temp is not None)
        if useMask:
            mask = self.par_mask

        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Initial voltage drops across the positive and negative devices
        dV0 = ncp.tile(vector,(matrix_pos.shape[0],1))

        # FOR A THREE-TERMINAL DEVICE:
        # If input is zero, the device is gated off and cannot conduct current regardless of the drain voltage
        # Ensure this by zeroing out the conductances where the row is not activated
        active_inputs = (ncp.abs(dV0) > 1e-9)
        matrix_pos *= active_inputs
        matrix_neg *= active_inputs
        Ires = (matrix_pos - matrix_neg)*dV0

        # Initial estimate of device currents
        # Initial dV is the same for positive and negative other than the sign
        dV_pos = dV0.copy()

        # Iteratively calculate parasitics and update device currents
        while Verr > Verr_th and Niters < Niters_max:
            # Calculate parasitic voltage drops
            if useMask:
                Isum_col = mask*ncp.cumsum(Ires,1)
            else:
                Isum_col = ncp.cumsum(Ires,1)

            Vpar = Rp*ncp.cumsum(Isum_col[:,::-1],1)[:,::-1]

            # Calculate the error for the current estimate of memristor currents
            VerrMat_pos = dV0 - Vpar - dV_pos

            # Evaluate overall error; if using SIMD, make sure only to count the cells that matter
            if useMask:
                Verr = ncp.max(ncp.abs(VerrMat_pos[mask]))
            else:
                Verr = ncp.max(ncp.abs(VerrMat_pos))

            if Verr < Verr_th:
                break

            # Update cell currents for the next iteration
            dV_pos += gamma*VerrMat_pos
            dV_neg = 2*dV0 - dV_pos
            Ires = matrix_pos*dV_pos - matrix_neg*dV_neg

            Niters += 1

        # Calculate the summed currents on the columns
        Icols = ncp.sum(Ires,axis=1)

        # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
        if Verr > Verr_th:
            raise RuntimeError('Parasitic resistance too high: could not converge!')
        del Ires
        return Icols


    def update_matrix(self, row_vector, col_vector, learning_rate, core_ind, randRecord=1e20):
        """
        randRecord : random value (between 0 and 1) passed in externally to determine whether weight
        update statistics will be collected on this training example
        """
        row_vector,col_vector =  self.clip_and_quantize_update_matrix_inputs(row_vector, col_vector)
        if self.params.numeric_params.Rp > 0 and self.params.numeric_params.parasitic_backprop:
            # Calculate outer product update in the presence of parasitics (experimental feature!!)
            update = self.calc_OPU_parasitics(row_vector*learning_rate,col_vector)
        else:
            update = ncp.outer(row_vector*learning_rate, col_vector)

        # If executing in SIMD mode, add together the updates in a batch
        if self.params.numeric_params.Nex_par > 1:
            Nx, Ny = self.matrix.shape
            update_large = update.copy()
            update = ncp.zeros((Nx,Ny),dtype=self.matrix.dtype)
            for m in range(self.params.numeric_params.Nex_par):
                x_start, x_end = m*Nx, (m+1)*Nx
                y_start, y_end = m*Ny, (m+1)*Ny
                update += update_large[x_start:x_end,y_start:y_end]

        target_update = update.copy()
        # check what type of update model is being used
        if self.params.numeric_params.update_model==UpdateModelEnum.ANALYTIC:
            # compute nonlinear update if needed
            if bool(self.params.numeric_params.nonlinearity.alpha):
                update = self._compute_nonlinear_update(self.matrix, update)
            # ***** check for write noise and apply
            if bool(self.params.numeric_params.write_noise.sigma):
                noise = self._compute_write_noise(update)
                # add noise to the update matrix
                update += noise
                
        # if using lookup table update
        elif self.params.numeric_params.update_model==UpdateModelEnum.DG_LOOKUP:
            update = self.params.numeric_params.dG_lookup.compute_update(update, self.matrix)
        else:
            raise ValueError("Unknown update model")

        # Save target update vs actual update, to be plotted later
        if self.record_updates and randRecord < 1e6/self.Nupdates_total and \
            (self.target_updates is None or len(self.target_updates) < self.Nupdates_total):
            if self.subtract_current_in_offset_xbar:
                target_update = target_update[1:,1:].flatten()
                real_update = update[1:,1:].flatten()
            else:
                real_update = update.copy().flatten()
            if self.target_updates is None:
                self.target_updates = target_update
                self.real_updates = real_update
            else:
                self.target_updates = np.concatenate((self.target_updates,target_update))
                self.real_updates = np.concatenate((self.real_updates,real_update))
            if len(self.target_updates) > self.Nupdates_total:
                self.target_updates = self.target_updates[:self.Nupdates_total]
                self.real_updates = self.real_updates[:self.Nupdates_total]

        if DEBUG_NUMERIC:
            print(row_vector)
            print('{oprod}')
            print(col_vector)
            print('x '+str(learning_rate))
            print('=')
            print(update)
            print('...')
            print(self.matrix + update)

        self.matrix += update

        # apply postpocessing / clip matrix values to limits and quantize
        self.matrix=self.xbar_params.weights.quantize(self.xbar_params.weight_clipping.clip(self.matrix))

        # Update matrix_temp:
        if self.params.numeric_params.Nex_par > 1:
            Nx,Ny = self.matrix.shape
            for m in range(self.params.numeric_params.Nex_par):
                x_start, x_end = m*Nx, (m+1)*Nx
                y_start, y_end = m*Ny, (m+1)*Ny
                self.matrix_temp[x_start:x_end,y_start:y_end] = self.matrix.copy()


    def calc_OPU_parasitics(self,row_vector,col_vector):
        """
        Calculates the result of an OPU in the presence of parasitics and select devices
        Performs the OPU in four phases based on the signs of x and delta: ++, +-, -+, --
        If balanced core, a separate four-phase update should be also be performed on the other core
        Note: for OPU only! Do not use if programming is serial (as in update_matrix_burst)

        If SIMD is enabled (Nex_par > 1), row_vector and col_vector should be expanded

        Note: this is still considered an experimental feature that assumes a specific circuit and
        electrical biasing scheme
        """
        Nex_par = self.params.numeric_params.Nex_par

        if Nex_par > 1:
            W0 = self.matrix_temp.copy()
        else:
            W0 = self.matrix.copy()

        dW = ncp.zeros(W0.shape,dtype=self.matrix.dtype)
        VrowS0 = self.params.numeric_params.circuit.VrowS
        VrowUS0 = self.params.numeric_params.circuit.VrowUS
        VcolUS0 = self.params.numeric_params.circuit.VcolUS
        Vselect = self.params.numeric_params.circuit.Vselect
        Vprog = self.params.numeric_params.circuit.Vprog

        # Scale to a max value of 1 (on one side) for both x and delta
        # separateScale: scale each example separately or scale all examples in a batch together
        #   Enabling this makes results a tiny bit more accurate, but incurs a very small overhead
        separateScale = True

        x = row_vector.copy()
        if Nex_par == 1 or not separateScale:
            f_col = np.max(np.abs(col_vector))
            delta = col_vector/f_col
        else:
            delta = col_vector.copy()
            Ndelta = len(col_vector)//Nex_par
            f_cols = np.zeros(Nex_par)
            for k in range(Nex_par):
                delta_k = col_vector[k*Ndelta:(k+1)*Ndelta]
                f_cols[k] = np.max(np.abs(delta_k))
                delta[k*Ndelta:(k+1)*Ndelta] = delta_k / f_cols[k]

        # These are the row and column vectors in voltage units
        if self.params.numeric_params.useGPU:
            x, delta = cp.asarray(row_vector), cp.asarray(delta)

        # Four-phase update
        for phase in range(4):
            # Switch circuit polarity for negative weight updates
            if phase == 1 or phase == 2:
                VrowS, VrowUS, VcolUS = -VrowS0, -VrowUS0, -VcolUS0
            else:
                VrowS, VrowUS, VcolUS = VrowS0, VrowUS0, VcolUS0

            # Determine fixed row voltages
            if phase == 0 or phase == 1:
                Trows = x*(x>0)
                Vrows = VrowS*(x>0) + VrowUS*(x<=0)
            elif phase == 2 or phase == 3:
                Trows = -x*(x<0)
                Vrows = VrowS*(x<0) + VrowUS*(x>=0)
            Trows = Trows.reshape(-1,1)

            # Variable column voltages
            if phase == 0:
                Vcols = (VrowS-Vselect-(delta+Vprog))*(delta>0) + VcolUS*(delta<=0)
            elif phase == 1:
                Vcols = (VrowS+Vselect-(delta-Vprog))*(delta<0) + VcolUS*(delta>=0)
            elif phase == 2:
                Vcols = (VrowS+Vselect+(delta+Vprog))*(delta>0) + VcolUS*(delta<=0)
            elif phase == 3:
                Vcols = (VrowS-Vselect+(delta-Vprog))*(delta<0) + VcolUS*(delta>=0)

            # Calculate voltage drops across memristors
            solved = False
            while not solved:
                solved = True
                try:
                    dV = self.calc_dV_OPU(Vrows,Vcols,W0)
                except ValueError:
                    self.params.numeric_params.convergence_param_opu *= 0.98
                    if self.params.numeric_params.convergence_param_opu <= 1e-2:
                        print("OPU failed to converge")
                    print("Reduced OPU convergence param to: "+str(self.params.numeric_params.convergence_param_opu))
                    solved = False

            # Calculate the voltage portion that goes into programming
            dV = (dV - Vprog)*(dV > Vprog) + (dV + Vprog)*(dV < -Vprog) 

            # Integrate over row pulse duration
            dW += dV*Trows

        # Convert back to normalized units
        if Nex_par > 1 and separateScale:
            for k in range(Nex_par):
                dW[:,k*Ndelta:(k+1)*Ndelta] *= f_cols[k] 
        else:
            dW = dW * f_col

        return dW


    def calc_dV_OPU(self,Vrows,Vcols,matrix):
        """
        Perform pseudo-circuit simulation of a single update step, given row and column voltages
        Returns the voltage drop across every array element (memristor + select device)
        """
        # Rp : parasitic resistance between two adjacent devices on a row or column
        # Since NumericCore does not know the resistance values of the devices, express Rp normalized
        # to the total conductance range (for example, 0.001)
        Rp = self.params.numeric_params.Rp
        Niters_max = self.params.numeric_params.Niters_max_parasitics
        Verr_th = self.params.numeric_params.Verr_th_opu
        Nex_par = self.params.numeric_params.Nex_par
        Vselect = self.params.numeric_params.circuit.Vselect
        gamma = self.params.numeric_params.convergence_param_opu

        if Nex_par > 1:
            mask = self.par_mask.copy()

        # Initial estimate of memristor currents
        # Use the memristor device voltage drops to calculate currents in the array
        if self.params.numeric_params.useGPU:
            # Unfortunately, CuPy does not have an outer method
            dV0 = cp.transpose(cp.tile(Vrows,(len(Vcols),1))) - cp.tile(Vcols,(len(Vrows),1))
        else:
            dV0 = np.subtract.outer(Vrows,Vcols)
        dV_res = (dV0 - Vselect)*(dV0 > Vselect) + (dV0 + Vselect)*(dV0 < -Vselect) 
        Ires = matrix*dV_res
        dV_prev = dV0.copy()

        # Initialize error and number of iterations
        Verr, Niters = 1e9, 0

        # Iteratively calculate parasitics and update memristor currents
        while Verr > Verr_th and Niters < Niters_max:

            if Nex_par > 1:
                Isum_row = mask*ncp.cumsum(Ires[::-1],0)[::-1]
                Isum_col = mask*ncp.cumsum(Ires,1)
            else:
                Isum_row = ncp.cumsum(Ires[::-1],0)[::-1]
                Isum_col = ncp.cumsum(Ires,1)
            Vdrops_row = Rp*ncp.cumsum(Isum_row,0)
            Vdrops_col = Rp*ncp.cumsum(Isum_col[:,::-1],1)[:,::-1]
            Vpar = Vdrops_col + Vdrops_row

            # Calculate the error for the current estimate of memristor currents
            Verr = ncp.mean(ncp.abs(dV0 - Vpar - dV_prev))
            if Verr < Verr_th:
                break

            dV = dV0 - Vpar
            dV = dV_prev - gamma*(dV_prev - dV)
            dV_res = (dV - Vselect)*(dV > Vselect) + (dV + Vselect)*(dV < -Vselect) 
            Ires = matrix*dV_res

            dV_prev = dV.copy()
            Niters += 1

        if Verr > Verr_th:
            raise ValueError('Parasitic resistance too high: could not converge!')

        return dV_res


    def update_matrix_burst(self, update_matrix, learning_rate, core_ind, randRecord=1e20):
        """
        Perform an update that is not described by an outer product, but otherwise same as update_matrix
        update_matrix is the desired update not including the learning rate
        PARASITICS ARE IGNORED: since programming is serial, the existing parasitic method tailored to OPU is not relevant for this
        Note: the update is not clipped or quantized, because it is undetermined how that should be handled in this physical situation
        """
        # use the row update param object to clip and quantize
        # this should work since the default xbar param row/col update range is (-1,1) which will be the same as the range of the product
        update_matrix = self.xbar_params.row_update.clip_and_quantize(update_matrix)
        update = learning_rate * update_matrix

        target_update = update.copy()
        # check what type of update model is being used
        if self.params.numeric_params.update_model==UpdateModelEnum.ANALYTIC:
            # compute nonlinear update if needed
            if bool(self.params.numeric_params.nonlinearity.alpha):
                update = self._compute_nonlinear_update(self.matrix, update)
            # ***** check for write noise and apply
            if bool(self.params.numeric_params.write_noise.sigma):
                noise = self._compute_write_noise(update)
                # add noise to the update matrix
                update += noise

        # if using lookup table update
        elif self.params.numeric_params.update_model==UpdateModelEnum.DG_LOOKUP:

            update = self.params.numeric_params.dG_lookup.compute_update(update,self.matrix)

        else:
            raise ValueError("Unknown update model")

        # Save target update vs actual update, to be plotted later
        if self.record_updates and randRecord < 1e6/self.Nupdates_total  and \
            (self.target_updates is None or len(self.target_updates) < self.Nupdates_total):
            if self.subtract_current_in_offset_xbar:
                target_update = target_update[1:,1:]
                real_update = update[1:,1:]
            else:
                real_update = update.copy()
            target = target_update.flatten()
            real = real_update.flatten()
            if self.target_updates is None:
                self.target_updates = target
                self.real_updates = real
            else:
                self.target_updates = np.concatenate((self.target_updates,target))
                self.real_updates = np.concatenate((self.real_updates,real))

            if len(self.target_updates) > self.Nupdates_total:
                self.target_updates = self.target_updates[:self.Nupdates_total]
                self.real_updates = self.real_updates[:self.Nupdates_total]

        if DEBUG_NUMERIC:
            print(update_matrix)
            print('{oprod}')
            print(col_vector)
            print('x '+str(learning_rate))
            print('=')
            print(update)
            print('...')
            print(self.matrix + update)

        self.matrix += update

        # apply postpocessing / clip matrix values to limits and quantize
        self.matrix=self.xbar_params.weights.quantize(self.xbar_params.weight_clipping.clip(self.matrix))


    def _read_matrix(self):
        print_debug_calls('NumericCore._read_matrix',self.matrix)
        return self.matrix.copy()

    
    def _save_matrix(self):
        return self.matrix.copy()


    def _restore_matrix(self, matrix):
        self.matrix = matrix.copy()


    # *********************************** functions to compute noise/ nonlinearities


    def read_noise_matrix(self):
        """
        Applies noise to a weight matrix, accounting for whether the matrix inclues replicated weights
        """
        if self.params.weight_error_params.noise_model == "none":
            return self.matrix

        if self.params.weight_error_params.noise_model == "alpha" and self.params.numeric_params.read_noise.sigma == 0:
            return self.matrix

        Ncopy = self.params.numeric_params.x_par * self.params.numeric_params.y_par
        
        # If doing a circuit simulation, must keep the full sized (sparse) matrix
        # Nex_par > 1 only if Rp > 0
        if self.params.numeric_params.Nex_par > 1:
            noisy_matrix = self._apply_read_noise(self.matrix)
            noisy_matrix *= self.par_mask

        # No parasitic resistance
        else:
            if Ncopy > 1:
                noisy_matrix = self._apply_read_noise(self.matrix_dense)
                Nx, Ny = self.matrix_temp.shape
                for m in range(Ncopy):
                    x_start, y_start = m*Nx, m*Ny
                    x_end, y_end = x_start+Nx, y_start+Ny
                    self.matrix[x_start:x_end,y_start:y_end] = noisy_matrix[m,:,:]
                noisy_matrix = self.matrix

            else:
                noisy_matrix = self._apply_read_noise(self.matrix)

        return noisy_matrix


    def _apply_read_noise(self,matrix):
        """
        returns a matrix with read noise applied
            creates new matrix if noise is added, else returns input
        :param matrix: matrix to apply noise to
        :return: noisy matrix
        """
        noisy_matrix = matrix.copy()
        if self.params.weight_error_params.noise_model == "alpha" and self.params.numeric_params.read_noise.sigma > 0:
            noisy_matrix = self.params.numeric_params.read_noise.apply(noisy_matrix, self.xbar_params.weights, noise_model=self.params.weight_error_params.noise_model)
        if self.params.weight_error_params.noise_model != "none" and self.params.weight_error_params.noise_model != "alpha":
            noisy_matrix = self.params.numeric_params.read_noise.apply(noisy_matrix, self.xbar_params.weights, noise_model=self.params.weight_error_params.noise_model, std_matrix=self.std_matrix)
        return noisy_matrix

    def _apply_weight_errors(self,load_errors=False,save_errors=False):
        """
        returns a matrix with weight errors applied
            creates new matrix if error is applied, else returns input
        :param matrix: matrix to apply error to
        Beta feature:
        :load_errors: rather than generating new random errors, load weight errors from a file
            This is useful to simulate different parts of a large dataset in parallel with the same weight errors
        :save_errors: save the perturbed weight matrix to a file
        :return: noisy matrix
        """
        # This function applies weight quantization, programming errors, and state drift

        # Keep a global tracker for core id: used if saving or loading weight errors
        if load_errors or save_errors:
            load_dir = "./saved_weight_errors/"
            save_dir = "./saved_weight_errors/"
            global id_counter
            if 'id_counter' not in globals():
                id_counter = 0
                if load_errors:
                    print("Loading previously saved weight errors in " + load_dir)
                elif save_errors:
                    import os
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    print("Saving perturbed conductance matrices to "+save_dir)
            else:
                id_counter += 1

        # Programming error
        applyQuantization = (self.params.algorithm_params.weight_bits > 0)
        applyError = (self.params.weight_error_params.error_model != "none")
        applyDrift = (self.params.weight_error_params.T > 0 and self.params.weight_error_params.drift_model != "none")
        setReadNoise = (self.params.weight_error_params.noise_model != "none" and self.params.weight_error_params.noise_model != "alpha")

        if (applyQuantization or applyError or applyDrift) and not load_errors:

            noisy_matrix = self.matrix.copy()

            if applyQuantization:
                noisy_matrix = self.params.weight_error_params.applyWeightQuantization(noisy_matrix,self.xbar_params.weights)

            # Note: if applyDrift is True and error_model is a custom device, the following will do nothing
            if applyError:
                noisy_matrix = self.params.weight_error_params.applyProgrammingError(noisy_matrix,self.xbar_params.weights)

            if applyDrift:
                noisy_matrix = self.params.weight_error_params.applyDrift(noisy_matrix,self.xbar_params.weights)

            self.matrix = noisy_matrix

        if load_errors:

            noisy_matrix = cp.array(np.load(load_dir + "core"+str(id_counter)+".npy"))
            if noisy_matrix.shape != self.matrix.shape:
                raise ValueError("Loaded error matrix has incorrect shape")
            self.matrix = noisy_matrix

        if save_errors:
            np.save(save_dir + "core"+str(id_counter)+".npy",cp.asnumpy(noisy_matrix))

        if setReadNoise:
            self.std_matrix = self.params.weight_error_params.setCustomReadNoise(self.matrix,self.xbar_params.weights)


    def _compute_write_noise(self, update):
        """
        uses matrix value stored in self
        :param update:
        :return: a matrix with the computed write noise
        """

        write_noise = self.params.numeric_params.write_noise

        range = self.xbar_params.weights.range

        # apply different noise models
        if write_noise.write_noise_model == WriteNoiseModelEnum.G_INDEPENDENT:
            sigma = np.sqrt(np.abs(update))
            sigma*= np.sqrt(range)         *write_noise.sigma # split into two lines to limit matrix ops
        elif write_noise.write_noise_model == WriteNoiseModelEnum.G_PROPORTIONAL:
            sigma = np.sqrt(np.abs(update))*self.matrix
            sigma*= np.sqrt(range) /range *write_noise.sigma # split into two lines to limit matrix ops
        elif write_noise.write_noise_model == WriteNoiseModelEnum.G_INVERSE:
            sigma = np.sqrt(np.abs(update))/self.matrix
            sigma*= np.sqrt(range) *range *write_noise.sigma # split into two lines to limit matrix ops
        else:
            raise ValueError("undefined write noise model"+ str(write_noise.write_noise_model))

        noise = np.random.normal(scale=1, size=update.shape)
        # if the update is zero, set the write noise to zero
        noise[update==0]=0
        noise *= sigma

        return noise


    def _compute_nonlinear_update(self, matrix, update):
        """
        return nonlinear update and overwrite update

            alpha = degree of nonlinearity, 0 is linear, large is more nonlinear
            w = current weight value
            wmax, wmin = max, min weights - set to hard limit
            w0 = (wmax-wmin) / (1 - exp(-alpha))

            delta = update / (wmax-wmin)
            for delta = positive change in weight (or conductance?):
              prefactor = w0 + wmin - w
              nonlinear_update = prefactor * (1 - exp(-delta*alpha))

            for delta = negative change in weight (or conductance?):
              prefactor = wmax - w0  - w
              nonlinear_update = prefactor * (1 - exp(+delta*alpha))
        """

        alpha = self.params.numeric_params.nonlinearity.alpha
        wmax = self.params.xbar_params.weights.maximum
        wmin = self.params.xbar_params.weights.minimum
        range = self.params.xbar_params.weights.range
        delta = update / range

        if self.params.numeric_params.nonlinearity.symmetric is False:
            # nonlinear prefactor
            # w0 = (wmax-wmin) / (1.0 - np.exp(-alpha))
            w0 = self.params.numeric_params.nonlinearity.asym_w0

            # Numpy method (fast):  It's better to compute everything twice than iterate through the matrix!
            # delta = update[i][j] / (wmax-wmin)

            # ********* compute positive updates:
            # tmp = 1.0 - exp(-alpha*delta)  # structure this way to minimize matrix ops / memory reallocations
            tmp = (-alpha)*delta
            tmp = 1.0 - np.exp(tmp, out=tmp)

            # if delta[i][j] > 0.0: newdelta = (w0+wmin - matrix[i][j]) * tmp
            update = (w0+wmin)- matrix    # (w0 - woffset) - orig_weight
            update *= tmp       # * (1-e^(-delta/A)

            # ********** compute negative updates

            #tmp3 = 1.0 - exp(alpha*delta)
            tmp = (alpha)*delta
            tmp = 1.0 - np.exp(tmp, out=tmp)

            # else: newdelta = (wneg - matrix[i][j]) * tmp2 (or tmp3)
            update_neg = (wmax-w0) - matrix    # (wmax-w0) - orig_wt
            update_neg *= tmp       # * (1-e^(delta/A)

            # find indices of neg values
            deltaneg = delta <= 0.0             # find all deltas <= 0.0

            # overwrite negative updates
            update[deltaneg] = update_neg[deltaneg]     #Add the non-positive deltas

            return update
        else:  # symmetric nonlinearity model
            # A = (wmax-wmin) *(np.exp(alpha)+1)/(np.exp(alpha)-1)
            # B = -(wmax-wmin)/(np.exp(alpha)-1)+wmin
            A = self.params.numeric_params.nonlinearity.sym_A
            B = self.params.numeric_params.nonlinearity.sym_B

            update = A / (1 + np.exp(-2*alpha*delta)* (A/(matrix-B)-1))-matrix+B

            return update


    def nonlinear_update_scaling(self, matrix, update):
        """
        Returns what the the update should be given a desired update to account for nonlinearity
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the update scaling at each value in matrix
        :param update: desired updates
        :type update: float
        :return:
        """
        matrix = self.clip_matrix_nonlinear_update_scaling(matrix) #clip matrix

        update = update* np.ones(np.shape(matrix))  # expand scalar update to size of matrix
        if self.params.numeric_params.update_model==UpdateModelEnum.ANALYTIC:
            if bool(self.params.numeric_params.nonlinearity.alpha):
                warn("The nonlinear update scaling for the analytic model is computed inefficiently and may run slowly")

                nonzero_indicies = update!=0    # can only consider nonzero update values
                update_target = update[nonzero_indicies]
                nonzero_matrix = matrix[nonzero_indicies]

                # find best input updates to get desired update using iterative refinement / newton's method
                input_update = update_target
                actual_update = self._compute_nonlinear_update(nonzero_matrix, input_update)

                while np.any(np.abs((update_target - actual_update)) > np.abs(
                        update_target) * 1e-6):  # iteratively refine the estimate untilk it converges
                    input_update = input_update * (update_target / actual_update)
                    actual_update = self._compute_nonlinear_update(nonzero_matrix, input_update)
                # TODO: ****************************make this more effient (i.e. directly calculate for each nolinearity case)

                scaled_update = np.zeros_like(update)
                scaled_update[nonzero_indicies] = input_update

        elif self.params.numeric_params.update_model == UpdateModelEnum.DG_LOOKUP:
            scaled_update = np.zeros_like(update)  # set zero updates to zero to avoid divide by zero warning
            nonzero_indicies = update != 0
            nonzero_update = update[nonzero_indicies]
            row_dim = update.shape[0]
            col_dim = update.shape[1]
            ret_update = self.params.numeric_params.dG_lookup.compute_update(nonzero_update,matrix[nonzero_indicies], row_dim,col_dim,disable_writenoise=True)#disable_writenoise=False
            #print('Doing nonlinear update scaling')
            scaled_update[nonzero_indicies] = nonzero_update ** 2 / ret_update

            # if ret_update is not None:
            #     scaled_update[nonzero_indicies] = nonzero_update**2/ret_update
            #     #scaled_update[nonzero_indicies] = nonzero_update**2/self.params.numeric_params.dG_lookup.compute_update(nonzero_update,matrix[nonzero_indicies], disable_writenoise=True)
            # else:
            #     scaled_update[nonzero_indicies] = np.zeros(np.shape(nonzero_update))

        else:
            raise ValueError("Unknown update model")

        return scaled_update


    def expand_matrix(self,Ncopy,mode=0):
        """
        Makes a big matrix containing M copies of the weight matrix so that multiple VMMs can be computed in parallel, SIMD style
        Off-diagonal blocks of this matrix are all zero
        If noise is enabled, additionally create a third matrix that contains all the nonzero elements of this big matrix
        Intended for GPU use only

        mode:
        - 0 for inference: self.matrix becomes big matrix
        - 1 for training: self.matrix remains small matrix
        """
        # Keep a copy of original matrix, both for construction of the expanded matrix and as a backup for later restoration if needed

        Nx,Ny = self.matrix.shape

        if mode == 0:

            self.matrix_temp = self.matrix.copy()

            noReadNoise = (self.params.weight_error_params.noise_model == "none" or \
                (self.params.weight_error_params.noise_model == "alpha" and self.params.numeric_params.read_noise.sigma == 0))

            if noReadNoise:
                
                if self.params.numeric_params.weight_reorder:
                    Kx = self.params.convolution_parameters.Kx
                    Ky = self.params.convolution_parameters.Ky
                    Nic = self.params.convolution_parameters.Nic
                    Noc = self.params.convolution_parameters.Noc
                    stride = self.params.convolution_parameters.stride
                    x_par = self.params.numeric_params.x_par # parallel windows in x
                    y_par = self.params.numeric_params.y_par # parallel windows in y
                    x_par_in = (x_par-1)*stride + Kx
                    y_par_in = (y_par-1)*stride + Ky
                    self.matrix = ncp.zeros((x_par*y_par*Noc,x_par_in*y_par_in*Nic),dtype=self.matrix.dtype)
                    m = 0
                    for ix in range(x_par):
                        for iy in range(y_par):
                            for ixx in range(Kx):
                                for iyy in range(Ky):
                                    # 1: Which elements of the flattened input should be indexed for this 2D point?
                                    x_coord = stride*ix + ixx
                                    y_coord = stride*iy + iyy
                                    row_xy = x_coord*y_par_in + y_coord
                                    x_start = row_xy
                                    x_end = row_xy + Nic*x_par_in*y_par_in
                                    # 2: Which elements of the weight matrix are used for this point?
                                    Wx_coord = ixx*Ky + iyy
                                    W_start = Wx_coord
                                    W_end = Wx_coord + Nic*Kx*Ky
                                    y_start, y_end = m*Noc, (m+1)*Noc
                                    self.matrix[y_start:y_end,x_start:x_end:(x_par_in*y_par_in)] = self.matrix_temp[:,W_start:W_end:(Kx*Ky)].copy()
                            m += 1

                else:
                    self.matrix = ncp.zeros((Ncopy*Nx,Ncopy*Ny),dtype=self.matrix.dtype)
                    for m in range(Ncopy):
                        x_start, x_end = m*Nx, (m+1)*Nx
                        y_start, y_end = m*Ny, (m+1)*Ny
                        self.matrix[x_start:x_end,y_start:y_end] = self.matrix_temp.copy()

            else:
                self.matrix = ncp.zeros((Ncopy*Nx,Ncopy*Ny),dtype=self.matrix.dtype)
                self.matrix_dense = ncp.zeros((Ncopy,Nx,Ny),dtype=self.matrix.dtype)
                for m in range(Ncopy):
                    x_start, x_end = m*Nx, (m+1)*Nx
                    y_start, y_end = m*Ny, (m+1)*Ny
                    self.matrix[x_start:x_end,y_start:y_end] = self.matrix_temp.copy()
                    self.matrix_dense[m,:,:] = self.matrix_temp.copy()

        else:
            self.matrix_temp = ncp.zeros((Ncopy*Nx,Ncopy*Ny),dtype=self.matrix.dtype)

            for m in range(Ncopy):
                x_start, x_end = m*Nx, (m+1)*Nx
                y_start, y_end = m*Ny, (m+1)*Ny
                self.matrix_temp[x_start:x_end,y_start:y_end] = self.matrix.copy()


    def unexpand_matrix(self,mode=0):
        """
        Undo the expansion operation in expand_matrix
        Intended for GPU use only

        mode:
        - 0 for inference: self.matrix returns to small matrix
        - 1 for training: self.matrix_temp erased

        """
        if mode == 0:
            self.matrix = self.matrix_temp.copy()

        self.matrix_temp = None
        self.matrix_dense = None