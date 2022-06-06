#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys, io, os, time, random, pkg_resources
import matplotlib as mpl
import numpy as np
sys.path.append("..")
from cross_sim import Backprop, Parameters
from matplotlib import pyplot as plt
from helpers.variability_helper import variability_helper
from helpers.dataset_loaders import load_data_mnist, load_data_fashion_mnist
datapath = os.path.join(os.path.dirname(__file__), '..',"cross_sim","cross_sim","data")

class Logger(object):
    """
    object to allow stdout to be directed to both the terminal and a file simultaneously
    """
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        if filename:
            self.log = open(filename, "w",1) # the 1 makes it write after every newline, w means re-write the file (a would append)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except AttributeError:
            pass

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        try:
            self.log.flush()
        except AttributeError:
            pass
        self.terminal.flush()
        pass

    def change_file(self,filename):
        """
        changes the output file
        """
        try:
            self.log.close()
        except AttributeError:
            pass
        self.log = open(filename, "w",1) # the 1 makes it write after every newline, w means re-write the file (a would append)


def set_min_update_size(params, relative_min_update, learning_rate):
    """
    :param params: parameters object to update
    :type params: Parameters
    :param relative_min_update: minimum update size as a fraction of the weight range
    :param learning_rate: learning rate used in backprop
    :return: nothing, params object is updated
    """
    #check that bit settings compatible with setting a well defined min update
    if params.xbar_params.col_update.bits== 0 or params.xbar_params.row_update.bits== 0:
        raise ValueError("In order to set a minimum update size the A/D must have a fixed number of bits")

    if (params.xbar_params.col_update.sign_bit==False and params.xbar_params.col_update.maximum!=0 and params.xbar_params.col_update.minimum!=0) \
     or (params.xbar_params.row_update.sign_bit== False and params.xbar_params.row_update.maximum!=0 and params.xbar_params.row_update.minimum!=0):
        raise ValueError("Need equal sized bit quantization")

    if (params.xbar_params.col_update.maximum!=-params.xbar_params.col_update.minimum and params.xbar_params.col_update.maximum!=0 and params.xbar_params.col_update.minimum!=0):
        raise ValueError("Need equal sized bit quantization")
    if (params.xbar_params.row_update.maximum!=-params.xbar_params.row_update.minimum and params.xbar_params.row_update.maximum!=0 and params.xbar_params.row_update.minimum!=0):
        raise ValueError("Need equal sized bit quantization")

    #compute current minimum update
    dX = params.algorithm_params.row_update.range
    dY = params.algorithm_params.col_update.range
    W_range = params.algorithm_params.weights.range
    row_levels = params.xbar_params.row_update.levels - 1
    col_levels = params.xbar_params.col_update.levels - 1
    current_min_update = learning_rate*dX*dY/W_range/row_levels/col_levels

    # adjust desired minimum update size if using periodic carry for the lowest order digit
    if params.periodic_carry_params.use_periodic_carry==True:
        relative_min_update/=params.periodic_carry_params.number_base**(params.periodic_carry_params.cores_per_weight-1)

    scaling = np.sqrt(relative_min_update/current_min_update)

    # update dX and dY
    params.algorithm_params.row_update.maximum*=scaling
    params.algorithm_params.row_update.minimum*=scaling
    params.algorithm_params.col_update.maximum*=scaling
    params.algorithm_params.col_update.minimum*=scaling


class train_neural_net(object):
    """
    Creates a neural network training object that sets all the required parameters and runs the neural network training
    The results are saved to file

    """

    def __init__(self, outdir):
        """
        :param outdir: the directory in which to save all the simulation results
        :return:
        """
        self.outdir = outdir
        # create outdir if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.output_log  = Logger()
        sys.stdout = self.output_log


    def set_params(self,**kwargs):
        """
        Pass parameters using kwargs to allow for a general parameter dict to be used
        This function should be called before train and sets all parameters of the neural core simulator
        See below for definitions

        # the lookup table defines a large conductance range, but using only part of it may give better results
        # Gmin_relative and Gmax_relative define the fraction of the lookup table (25% to 75% of the range) to target using
        # weights can go outside the targeted range, but the weight scaling will be based on the targeted range.
        """
        #######################
        #### load relevant parameters from arg dict
        
        # Dataset
        task = kwargs.get("task","small")
        lookup_table = kwargs.get("lookup_table",None)
        rad_level = kwargs.get("rad_level",0)
        clipscale = kwargs.get("clipscale",1.5) # 0 means no clipping
        wtmodel = kwargs.get("wtmodel","BALANCED")
        learnbias = kwargs.get("learnbias",(False,False))

        # ENODe specific setting
        # Centered means all LUT's will be centered around the same conductance, using a resistor paired
        # with each device to eliminate the offsets. This leads to higher accuracy
        centered = kwargs.get("centered",True)

        # Gmin_relative: min of weight range to use (works with clipscale to set scaling)
        # Gmax_relative: max of weight range to use
        Gmin_relative = kwargs.get("Gmin_relative",0.1)
        Gmax_relative = kwargs.get("Gmax_relative",0.9)
        stochastic_quantization = kwargs.get("stochastic_quantization",False)

        # ADC/DAC settings: stochastic_updates enables stochastic rounding
        # Set a2dmodel to "NONE" to disable
        a2dmodel = kwargs.get("a2dmodel","SIGMOID")
        stochastic_updates = kwargs.get("stochastic_updates",False)

        # Diagnostic data collection settings: True/False and # updates to record during training
        diagnosticParams = kwargs.get("diagnosticParams",[False,1e10])

        Rp = kwargs.get("Rp",0)
        Nex_par = kwargs.get("Nex_par",1)
        useGPU = kwargs.get("useGPU",False)
        use_einsum = kwargs.get("use_einsum",False)
        vectorize = kwargs.get("vectorize",True)
        
        periodic_carry = kwargs.get("periodic_carry",False)
        pc_number_base = kwargs.get("pc_number_base",5)
        pc_Nslices = kwargs.get("pc_Nslices",3)
        
        Nlevels = kwargs.get("Nlevels",0) # Number of levels per device if quantizing

        ################  create parameter objects with all neural core settings for first core
        params = Parameters()

        params.algorithm_params.crossbar_type=wtmodel
        params.algorithm_params.sim_type="NUMERIC"
        params.xbar_params.weights.minimum = 0.1
        params.xbar_params.weights.maximum = 1
        params.algorithm_params.disable_clipping = False
        params.numeric_params.useEINSUM = use_einsum

        params.numeric_params.useGPU = useGPU

        # Periodic carry
        if periodic_carry:
            params.periodic_carry_params.use_periodic_carry = periodic_carry
            params.periodic_carry_params.cores_per_weight = pc_Nslices
            params.periodic_carry_params.number_base = pc_number_base
            params.periodic_carry_params.carry_threshold = 0.5
            params.periodic_carry_params.read_low_order_bits = True
            params.periodic_carry_params.carry_frequency = [10]
            params.periodic_carry_params.exact_carries = False

        # Parasitic resistance and write circuit simulation settings
        if Rp > 0: 
            Wavg = 0.5*(params.xbar_params.weights.minimum+params.xbar_params.weights.maximum)
            params.numeric_params.Rp = Rp/Wavg
            params.numeric_params.Niters_max_parasitics = 100
            params.numeric_params.Verr_th_mvm = 1e-3
            params.numeric_params.Verr_th_opu = 1e-3
            params.numeric_params.parasitic_backprop = False
            params.numeric_params.circuit.noRowParasitics = True
            params.numeric_params.circuit.VrowS = 1.0
            params.numeric_params.circuit.VrowUS = -0.3
            params.numeric_params.circuit.VcolUS = 0.5
            params.numeric_params.circuit.Vprog = 0.1333
            params.numeric_params.circuit.Vselect = 0.6667
            params.numeric_params.circuit.Vread = 0.7667
            params.numeric_params.circuit.mvm_encoding = "temporal"

            # Circuit solver convergence parameter: seems to be problem dependent
            # For MNIST inference
            if Rp/Wavg <= 1.5e-5:
                params.numeric_params.convergence_param = 0.5
                params.numeric_params.convergence_param_opu = 0.5
            elif Rp/Wavg <= 3.5e-5:
                params.numeric_params.convergence_param = 0.3
                params.numeric_params.convergence_param_opu = 0.3
            elif Rp/Wavg <= 7.5e-5:
                params.numeric_params.convergence_param = 0.2
                params.numeric_params.convergence_param_opu = 0.2
            else:
                params.numeric_params.convergence_param = 0.15
                params.numeric_params.convergence_param_opu = 0.15

        params.numeric_params.Nex_par = Nex_par
        if Nex_par > 1 and not wtmodel == "BALANCED":
            raise ValueError("Parallel VMM/MVM/OPU is only compatible with balanced core for now")
            
        # Diagnostics
        record_updates, Nupdates_total = diagnosticParams[0], diagnosticParams[1]
        params.analytics_params.record_updates = record_updates
        if record_updates:
            params.analytics_params.Nupdates_total = Nupdates_total

        if stochastic_updates:
            params.xbar_params.row_update.stochastic_rounding=True
            params.xbar_params.col_update.stochastic_rounding=True

        
        ###################################################
        #
        #   INPUT/OUTPUT QUANTIZATION AND RANGE
        #
        ###################################################

        # If the maximum update is larger than the weight range, large updates will be clipped

        if a2dmodel=="NONE":
            # set clipping parameters to give ideal accuracy
            params.algorithm_params.col_input.maximum = 100
            params.algorithm_params.col_input.minimum = -100
            params.xbar_params.col_input.bits = 0
            params.algorithm_params.row_input.maximum = 100
            params.algorithm_params.row_input.minimum = -100
            params.xbar_params.row_input.bits = 0
            params.algorithm_params.col_output.maximum = 100000
            params.algorithm_params.col_output.minimum = -100000
            params.xbar_params.col_output.bits = 0
            params.algorithm_params.row_output.maximum = 100000
            params.algorithm_params.row_output.minimum = -100000
            params.xbar_params.row_output.bits = 0
            params.algorithm_params.col_update.maximum = 100
            params.algorithm_params.col_update.minimum = -100
            params.xbar_params.col_update.bits = 0
            params.algorithm_params.row_update.maximum = 100
            params.algorithm_params.row_update.minimum = -100
            params.xbar_params.row_update.bits = 0

        elif a2dmodel=="SIGMOID" or a2dmodel=="RELU" or a2dmodel=="SOFTMAX_OUTPUT": 
            # Using calibrated bit precision/input ranges to minimize error
            # Forward prop
            params.xbar_params.col_input.bits = 7
            params.xbar_params.col_input.sign_bit = True  # total bits  =  bits + sign_bit
            # Back prop
            params.xbar_params.row_input.bits = 7
            params.xbar_params.row_input.sign_bit = True  # total bits  =  bits + sign_bit
            # Back prop
            params.xbar_params.col_output.bits = 7
            params.xbar_params.col_output.sign_bit = True  # total bits  =  bits + sign_bit
            # Forward prop
            params.xbar_params.row_output.bits = 7
            params.xbar_params.row_output.sign_bit = True  # total bits  =  bits + sign_bit
            # Update comes from row out passed through neuron
            params.xbar_params.row_update.bits = 6
            params.xbar_params.row_update.sign_bit = True  # total bits  =  bits + sign_bit
            # Backpropogated error
            params.xbar_params.col_update.bits = 4
            params.xbar_params.col_update.sign_bit = True  # total bits  =  bits + sign_bit

            if a2dmodel=="SIGMOID":
                # forward prop
                params.algorithm_params.col_input.maximum = 1
                params.algorithm_params.col_input.minimum = -1
                # back prop
                params.algorithm_params.row_input.maximum = 1
                params.algorithm_params.row_input.minimum = -1
                # cols used for backprop
                params.algorithm_params.col_output.maximum = 2 * 2
                params.algorithm_params.col_output.minimum = -2 * 2
                # rows used for forward propagation
                params.algorithm_params.row_output.maximum = 4 * 1.5
                params.algorithm_params.row_output.minimum = -4 * 1.5
                # update comes from row out passed through neuron
                params.algorithm_params.col_update.maximum = 1
                params.algorithm_params.col_update.minimum = -1
                # backpropogated error
                params.algorithm_params.row_update.maximum = 0.25 * params.algorithm_params.col_output.maximum * 0.1
                params.algorithm_params.row_update.minimum = -0.25 * params.algorithm_params.col_output.maximum * 0.1

            elif a2dmodel=="RELU":
                # forward prop
                params.algorithm_params.col_input.maximum = 10
                params.algorithm_params.col_input.minimum = -10
                # rows used for forward propagation
                params.algorithm_params.row_output.maximum = 20
                params.algorithm_params.row_output.minimum = -20
                # back prop
                params.algorithm_params.row_input.maximum = 1
                params.algorithm_params.row_input.minimum = -1
                # cols used for backprop
                params.algorithm_params.col_output.maximum = 3
                params.algorithm_params.col_output.minimum = -3
                # update comes from row out passed through neuron
                params.algorithm_params.col_update.maximum = 10
                params.algorithm_params.col_update.minimum = -10
                #backpropogated error
                params.algorithm_params.row_update.maximum = 3
                params.algorithm_params.row_update.minimum = -3

        ###################################################
        #
        #   LOOKUP TABLE SETTINGS
        #
        ###################################################

        # Lookup table option
        params.numeric_params.update_model="DG_LOOKUP"
        params.numeric_params.dG_lookup.gpu_engaged = useGPU
        params.numeric_params.dG_lookup.vectorize = vectorize

        # Multi-lookup table cases can be handled together
        if lookup_table in ("ENODe_multi",
            "TaOx_large_set",
            "TaOx_medium_set",
            "TaOx_small_set",
            "DWMTJ_STT_0K_multi",
            "DWMTJ_STT_300K_multi",
            "DWMTJ_STT_400K_multi",
            "DWMTJ_SOT_0K_multi",
            "DWMTJ_SOT_300K_multi",
            "DWMTJ_SOT_400K_multi"):

            if lookup_table == "ENODe_multi":
                num_luts = 9
                table_dim = 501
                if centered:
                    lut_folder = "ENODe_multi_centered/"
                else:
                    lut_folder = "ENODe_multi_uncentered/"

            elif "TaOx" in lookup_table:
                table_dim = 501
                lut_folder = "TaOx_all/"
                if lookup_table == "TaOx_large_set":
                    num_luts = 41
                    lut_indices = np.arange(num_luts)
                elif lookup_table == "TaOx_medium_set":
                    lut_indices = np.array([0,2,4,6,7,9,10,11,12,13,15,18,20,21,23,24,27,30,33,35,36,37,39,40])
                    num_luts = len(lut_indices)
                elif lookup_table == "TaOx_small_set":
                    lut_indices = np.array([2,4,6,7,9,10,11,12,13,15,18])
                    num_luts = len(lut_indices)

            elif "DWMTJ_STT" in lookup_table:
                num_luts = 20
                table_dim = 501
                if "_0K" in lookup_table:
                    lut_folder = "DWMTJ_STT/0K/"
                elif "300K" in lookup_table:
                    lut_folder = "DWMTJ_STT/300K/"
                elif "400K" in lookup_table:
                    lut_folder = "DWMTJ_STT/400K/"
            elif "DWMTJ_SOT" in lookup_table:
                num_luts = 20
                table_dim = 501
                if "_0K" in lookup_table:
                    lut_folder = "DWMTJ_SOT/0K/"
                elif "300K" in lookup_table:
                    lut_folder = "DWMTJ_SOT/300K/"
                elif "400K" in lookup_table:
                    lut_folder = "DWMTJ_SOT/400K/"

            cond_metamat_incr = np.zeros((table_dim,table_dim,num_luts))
            cond_metamat_decr = np.zeros((table_dim,table_dim,num_luts))
            dw_error_dec_lutmat =np.zeros((table_dim,table_dim,num_luts))
            dw_error_inc_lutmat =np.zeros((table_dim,table_dim,num_luts))
            gbins_metamat = np.zeros((table_dim,num_luts))
            ave_dw_dec_lutmat = np.zeros(num_luts)
            ave_dw_inc_lutmat = np.zeros(num_luts)
            w_steps_vec = np.zeros(num_luts)
            ave_dw_dec_vec_lutmat =np.zeros((table_dim,num_luts))
            ave_dw_inc_vec_lutmat =np.zeros((table_dim,num_luts))
            CDF_vec_lutmat = np.zeros((table_dim,num_luts))

            # Retrieve file paths
            inc_files = [None for i in range(num_luts)]
            dec_files = [None for i in range(num_luts)]

            # For DW, also save the G values corresponding to the equally spaced notch positions
            G_limits = np.zeros((num_luts,2))

            for i in range(num_luts):
                if lookup_table == "ENODe_multi":
                    inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+str(i+1)+"/dG_increasing"+str(i)+".txt")
                    dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+str(i+1)+"/dG_decreasing"+str(i)+".txt")
                elif "DWMTJ" in lookup_table:
                    inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+"dG_increasing"+str(i)+".txt")
                    dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+"dG_decreasing"+str(i)+".txt")       
                elif "TaOx" in lookup_table:
                    inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+str(lut_indices[i]+1)+"/dG_increasing.txt")
                    dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/"+lut_folder+str(lut_indices[i]+1)+"/dG_decreasing.txt")
                
                G_vec_inc = np.genfromtxt(inc_file,delimiter=',',skip_header=1, max_rows=1)  # max_rows option requires numpy v1.10 or higher
                gbins_metamat[:,i] = G_vec_inc
                inc_files[i] = inc_file
                dec_files[i] = dec_file

                if Nlevels > 0:
                    G_limits[i,:] = np.array([np.min(G_vec_inc),np.max(G_vec_inc)])                


            Gmax_global_std = np.amax(gbins_metamat)
            Gmin_global_std = np.amin(gbins_metamat)
            print("Global max conductance = {:.4f}".format(Gmax_global_std*1e6)+" uS")
            print("Global min conductance = {:.4f}".format(Gmin_global_std*1e6)+" uS")
            Gmax = Gmin_global_std + (Gmax_global_std-Gmin_global_std)*Gmax_relative
            Gmin = Gmin_global_std + (Gmax_global_std-Gmin_global_std)*Gmin_relative
            print("The lookup table range used is {:.4f}".format(Gmin*1e6)+" uS to {:.4f}".format(Gmax*1e6)+" uS. \nThe xbar weight limits will be rescaled accordingly")
            params.xbar_params.weight_clipping.minimum = Gmin_global_std/Gmax
            params.xbar_params.weight_clipping.maximum = Gmax_global_std/Gmax
            params.xbar_params.weights.minimum = Gmin/Gmax
            params.xbar_params.weights.maximum = 1
            gbins_metamat = gbins_metamat/Gmax
            w_steps = np.diff(gbins_metamat,axis=0)
            w_step = np.mean(w_steps,axis=0)
            w_steps_vec = w_step

            for i in range(num_luts):
                inc_file, dec_file = inc_files[i], dec_files[i]
                CDF_vec_inc = np.genfromtxt(inc_file,delimiter=',',skip_header=2, max_rows=1)  # max_rows option requires numpy v1.10 or higher
                CDF_vec_lutmat[:,i] = CDF_vec_inc
                G_vec_inc = np.genfromtxt(inc_file,delimiter=',',skip_header=1, max_rows=1)  # max_rows option requires numpy v1.10 or higher
                dG_matrix_inc = np.genfromtxt(inc_file,delimiter=',',skip_header=3)
                dG_matrix_decr = np.genfromtxt(dec_file,delimiter=',',skip_header=3)
                dw_matrix_inc = dG_matrix_inc/Gmax
                dw_matrix_dec = dG_matrix_decr/Gmax
                cond_metamat_incr[:,:,i] = dw_matrix_inc #fill - is this really needed
                cond_metamat_decr[:,:,i] = dw_matrix_dec #fill - is this really needed
                ave_dw_inc, ave_dw_dec, ave_dw_inc_vec, ave_dw_dec_vec, dw_error_matrix_dec,dw_error_matrix_inc = variability_helper(i, inc_file, dec_file, Gmax, CDF_vec_inc,G_vec_inc,dG_matrix_inc,dG_matrix_decr)
                dw_error_dec_lutmat[:,:,i] = dw_error_matrix_dec #fill
                dw_error_inc_lutmat[:,:,i] =  dw_error_matrix_inc #fill
                ave_dw_dec_vec_lutmat[:,i] = ave_dw_dec_vec #fill
                ave_dw_inc_vec_lutmat[:,i] = ave_dw_inc_vec #fill
                ave_dw_dec_lutmat[i] = ave_dw_dec #fill
                ave_dw_inc_lutmat[i] = ave_dw_inc #fill

            # Exposing these arrays so numeric_params can use them later on
            params.numeric_params.dG_lookup.file_decreasing = cond_metamat_decr
            params.numeric_params.dG_lookup.file_increasing = cond_metamat_incr
            params.numeric_params.dG_lookup.gbinsmat = gbins_metamat
            params.numeric_params.dG_lookup.CDF_vec_lutmat = CDF_vec_lutmat
            params.numeric_params.dG_lookup.ave_dw_dec_vec_lutmat = ave_dw_dec_vec_lutmat
            params.numeric_params.dG_lookup.ave_dw_inc_vec_lutmat = ave_dw_inc_vec_lutmat
            params.numeric_params.dG_lookup.ave_dw_dec_lutmat = ave_dw_dec_lutmat
            params.numeric_params.dG_lookup.ave_dw_inc_lutmat = ave_dw_inc_lutmat
            params.numeric_params.dG_lookup.w_steps_lut_vec = w_steps_vec
            params.numeric_params.dG_lookup.dw_error_dec_lutmat = dw_error_dec_lutmat
            params.numeric_params.dG_lookup.dw_error_inc_lutmat = dw_error_inc_lutmat
            params.numeric_params.dG_lookup.multi_LUT = True
            
            if Nlevels > 0:
                params.numeric_params.dG_lookup.Nlevels = Nlevels
                params.numeric_params.dG_lookup.w_limits = G_limits / Gmax # normalize
                params.numeric_params.dG_lookup.stochastic_quantization = stochastic_quantization

        elif lookup_table == "ENODe":
            inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/ENODe/dG_increasing.txt")
            dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/ENODe/dG_decreasing.txt")
            if os.path.isfile(inc_file) and os.path.isfile(dec_file):
                params.numeric_params.dG_lookup.file_decreasing = dec_file
                params.numeric_params.dG_lookup.file_increasing = inc_file
            
        elif lookup_table == "TaOx":
            inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/TaOx/dG_increasing.txt")
            dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/TaOx/dG_decreasing.txt")
            if os.path.isfile(inc_file) and os.path.isfile(dec_file):
                params.numeric_params.dG_lookup.file_decreasing = dec_file
                params.numeric_params.dG_lookup.file_increasing = inc_file

        elif lookup_table is not None and "DWMTJ" in lookup_table and "multi" not in lookup_table:
            if "STT" in lookup_table:
                device = "STT"
            elif "SOT" in lookup_table:
                device = "SOT"
            if "_0K" in lookup_table:
                inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/0K/dG_increasing0.txt")
                dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/0K/dG_decreasing0.txt")
            elif "300K" in lookup_table:
                inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/300K/dG_increasing0.txt")
                dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/300K/dG_decreasing0.txt")
            elif "400K" in lookup_table:
                inc_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/400K/dG_increasing0.txt")
                dec_file = pkg_resources.resource_filename("cross_sim","data/lookup_tables/DWMTJ_"+device+"/400K/dG_decreasing0.txt")
            if os.path.isfile(inc_file) and os.path.isfile(dec_file):
                params.numeric_params.dG_lookup.file_decreasing = dec_file
                params.numeric_params.dG_lookup.file_increasing = inc_file

        elif lookup_table is None:
            params.numeric_params.update_model = "ANALYTIC"

        elif os.path.isdir(lookup_table):
            params.numeric_params.dG_lookup.file_decreasing = os.path.join(lookup_table, 'dG_decreasing.txt')
            params.numeric_params.dG_lookup.file_increasing = os.path.join(lookup_table, 'dG_increasing.txt')

        else:
            lookup_dir = os.path.join(os.path.join(datapath, 'lookup_tables') , lookup_table)
            inc_file = os.path.join(lookup_dir, 'dG_increasing.txt')
            dec_file = os.path.join(lookup_dir, 'dG_decreasing.txt')
            if os.path.isfile(inc_file) and os.path.isfile(dec_file):
                params.numeric_params.dG_lookup.file_decreasing = dec_file
                params.numeric_params.dG_lookup.file_increasing = inc_file
            else:
                raise ValueError("Undefined Lookup Table Model "+str(lookup_table))

        params.numeric_params.dG_lookup.Gmin_relative = Gmin_relative
        params.numeric_params.dG_lookup.Gmax_relative = Gmax_relative

        # set matrix clipping limits to clipscale (limits are further scaled in train)
        params.algorithm_params.weights.maximum = clipscale
        params.algorithm_params.weights.minimum = -clipscale

        return params


    def train(self, filename, dataset, params=None,  params2=None, perceptron=False, **kwargs):
        """
        This function trains a neural network on 1 of 3 datasets, using the lookup_table specified by lookup_table and saves the results.
        Currently able to train a neural net with either one core (perceptron) or two cores (two-layer perceptron)

        :param filename: what file to save the simulation results to
        :param dataset: what dataset to use.  The weight ranges are scaled based on the dataset.
        :param params:  a parameters object containing all the settings for the neural core.  If none, a numeric simulation is run.
                        The weight ranges are further adjusted based on the dataset, unless scale_weights=False
        :param params2: if set, it contains settings to use for the second core of the neural net.  If none, settings from the first core are copied

        Keyword definitions are below.
        """
        ########## load values from dict

        # If true, weights are scaled based on dataset, if false weights are not scaled.
        scale_weights = kwargs.get("scale_weights",True)
        plot_weight_hist = kwargs.get("plot_weight_hist",False)
        n_epochs = kwargs.get("n_epochs",20)
        seed = random.randrange(1,1000000)
        # seed = 1000
        alpha = kwargs.get("alpha",0.1)
        min_update = kwargs.get("min_update",None)
        clip_numeric = kwargs.get("clip_numeric",False)
        verbose = kwargs.get("verbose",False)

        # MLP topology
        sizes = kwargs.get("sizes",None)

        # Activation function
        activate = kwargs.get("activate","SIGMOID")
        activate_output = kwargs.get("activate_output","SIGMOID")
        neuron_shift = kwargs.get("neuron_shift",0)

        # pos_init: initialize weights to positive values
        pos_init = kwargs.get("pos_init",False)
        learnbias = kwargs.get("learnbias",False)
        whetstone = kwargs.get("whetstone",False)
        lr_sched = kwargs.get("lr_sched",False)
        lr_sched_function = kwargs.get("lr_sched_function",None)

        batchsize = kwargs.get("batchsize",1)
        ntset = kwargs.get("ntset",0)
        ncset = kwargs.get("ncset",0)

        loadModelPath = kwargs.get("loadModelPath",None)
        saveModelPath = kwargs.get("saveModelPath",None)

        # Only used for numeric, for LUT the option is retrieved from params
        useGPU = kwargs.get("useGPU",False)

        ###### set the output filename
        self.output_log.change_file(os.path.join(self.outdir, filename) )

        ####################   backprop related parameters

        # optimal initial weight scaling and learning rate
        matscale = "bengio"

        # load appropriate dataset
        if dataset == "small":
            if sizes is None:
                sizes = (64,36,10)
            else:
                if sizes[0] != 64 or sizes[-1] != 10:
                    raise ValueError("MLP layer size not compatible with dataset")
            trainfile = pkg_resources.resource_filename("cross_sim", "data/datasets/small_digits/image_digit_small.train")
            testfile = pkg_resources.resource_filename("cross_sim", "data/datasets/small_digits/image_digit_small.test")
        elif dataset == "mnist":
            if sizes is None:
                sizes = (784,300,10)
            else:
                if sizes[0] != 784 or sizes[-1] != 10:
                    raise ValueError("MLP layer size not compatible with dataset")
            (x_train, y_train), (x_test, y_test) = load_data_mnist(training=True)
            data_train = np.empty((x_train.shape[0],x_train.shape[1]+1),dtype=np.float32)
            data_test = np.empty((x_test.shape[0],x_test.shape[1]+1),dtype=np.float32)
            data_train[:,:-1] = x_train
            data_train[:,-1] = y_train
            data_test[:,:-1] = x_test
            data_test[:,-1] = y_test
        elif dataset == "fashion":
            if sizes is None:
                sizes = (784,300,10)
            else:
                if sizes[0] != 784 or sizes[-1] != 10:
                    raise ValueError("MLP layer size not compatible with dataset")
            (x_train, y_train), (x_test, y_test) = load_data_fashion_mnist(training=True)
            x_train = x_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
        elif dataset =="cyber":
            if sizes is None:
                sizes = (256,150,9)
            else:
                if sizes[0] != 256 or sizes[-1] != 9:
                    raise ValueError("MLP layer size not compatible with dataset")
            trainfile = pkg_resources.resource_filename("cross_sim","data/datasets/file_types/cyber.train")
            testfile = pkg_resources.resource_filename("cross_sim","data/datasets/file_types/cyber.test")
        elif dataset =="iris": # use same data for test and train initially
            if sizes is None:
                sizes = (4,8,3)
            else:
                if sizes[0] != 4 or sizes[-1] != 3:
                    raise ValueError("MLP layer size not compatible with dataset")
            trainfile = pkg_resources.resource_filename("cross_sim","data/datasets/iris/bezdekIris.train")
            testfile = pkg_resources.resource_filename("cross_sim","data/datasets/iris/bezdekIris.test")
        elif dataset =="SPECTF": # use same data for test and train initially
            if sizes is None:
                sizes = (44,22,2)
            else:
                if sizes[0] != 44 or sizes[-1] != 2:
                    raise ValueError("MLP layer size not compatible with dataset")
            trainfile = pkg_resources.resource_filename("cross_sim", "data/datasets/SPECTF/SPECTF_original.train")
            testfile = pkg_resources.resource_filename("cross_sim","data/datasets/SPECTF/SPECTF_original.test")
        elif dataset == "UCI_HAR":
            if sizes is None:
                sizes = (561,200,6)
            else:
                if sizes[0] != 561 or sizes[-1] != 6:
                    raise ValueError("MLP layer size not compatible with dataset")
            trainfile = pkg_resources.resource_filename("cross_sim", "data/datasets/UCI_HAR/HAR.train")
            testfile = pkg_resources.resource_filename("cross_sim","data/datasets/UCI_HAR/HAR.test")
        else:
            raise ValueError("Unknown dataset "+str(dataset))

        # Add unused (x,y) dimensions to size vector
        sizes_new = [(1,1,size) for size in sizes]
        sizes = sizes_new

        # weight scaling different for datasets, calibrated to maximize dynamic range:
        if dataset == "small" or dataset =="small_reduced":
            baseline_mat = np.array([0.866, 1.93])
        elif dataset == "mnist":
            baseline_mat = np.array([0.218, 1.05])
        elif dataset == "fashion":
            if len(sizes) == 3:
                baseline_mat = np.array([0.7226, 0.8887])
            elif len(sizes) == 4:
                baseline_mat = np.array([0.7177, 0.7576, 0.8675])
        elif dataset =="cyber":
            baseline_mat = np.array([0.219, 0.444])
        elif dataset == "iris":
            baseline_mat = np.array([1.74761165803*2/1.5, 2.2352760706*2/1.5])
        elif dataset == "SPECTF":
            baseline_mat = np.array([0.264*2, 0.847*2])
        elif dataset == "or" or dataset == "xor" or dataset == "and" or dataset == "nand":
            baseline_mat = np.array([7 / params.algorithm_params.weights.maximum])
        elif dataset == "UCI_HAR":
            baseline_mat = np.array([1, 1])
        else:
            raise ValueError("Unknown dataset "+str(dataset))


        #######################

        time_start = time.time()
        Nlayers = len(sizes)-1        

        # intialize backprop simulation object
        bp = Backprop(sizes, seed=seed)

        for k in range(Nlayers-1):
            bp.set_activations(layer=k, style=activate, leakyslope=0, shift=neuron_shift) # set activation function type
        bp.set_activate_output(style=activate_output,shift = 0)
        bp.set_learning_rate(alpha, lr_sched, lr_sched_function)
        bp.layerTypes = ['dense'] * (Nlayers)

        if params is not None:
            useGPU = params.numeric_params.useGPU
        else:
            useGPU = useGPU
        bp.init_GPU(useGPU,0)

        # learnbias is either exactly specified for each layer, or the same value is used for all layers
        if type(learnbias) == bool:
            learnbias = [learnbias] * Nlayers
        elif len(learnbias) != Nlayers:
            learnbias = [learnbias[0]] * Nlayers

        ######## NUMERIC SIMULATION
        if params is None:
            model = "numeric"
            for k in range(Nlayers):
                num_core_style_k = ("old_bias" if learnbias[k] else "old")

                if clip_numeric:
                    bp.ncore(which=k+1,style=num_core_style_k,wtmodel="POSNEG",truncate=1,\
                        clip=baseline_mat[k]*1.5,useGPU=useGPU)
                else:
                    bp.ncore(which=k+1,style=num_core_style_k,wtmodel="POSNEG",truncate=0,\
                        useGPU=useGPU)

        ######## CROSSBAR SIMULATION
        else:
            model = ("analytic" if params.numeric_params.update_model == 0 else "lookup")
            paramsList = [None] * Nlayers

            for k in range(Nlayers):
                params_k = params.copy()
                
                # Set the array size used for multiLUT assignment matrices
                if model == "lookup" and params.numeric_params.dG_lookup.multi_LUT:
                    matsize = np.array((sizes[k+1][2], sizes[k][2]+int(learnbias[k]))).astype(int)
                    if params_k.algorithm_params.crossbar_type == 1: # offset core
                        matsize += 1
                    params_k.numeric_params.dG_lookup.matsize = matsize

                 # set matrix clipping limits
                if scale_weights:        
                    params_k.algorithm_params.weights.maximum *= baseline_mat[k]
                    params_k.algorithm_params.weights.minimum *= baseline_mat[k]
                    print("Matrix "+str(k+1)+" Weight Limit = ",params_k.algorithm_params.weights.maximum)
                
                if min_update is not None:
                    set_min_update_size(params_k, min_update, alpha)

                paramsList[k] = params_k
                lut_core_style_k = ("new_bias" if learnbias[k] else "new")
                bp.ncore(which=k+1,  style=lut_core_style_k, params=params_k)

        # load the training and the test data
        if dataset=="cyber" or dataset=="iris" or dataset=="SPECTF":  # need to scale cyber input data differently
            ntrain,indata_train,result_train,info = bp.read_inputs(trainfile,shuffle=1,scale="colgauss",scalereturn=1,verbose=False)
            ntest,indata_test,result_test = bp.read_inputs(testfile,scale="colgauss",scaleparam=info,verbose=False)
        elif dataset == "fashion":
            ntrain, ntest = x_train.shape[0], x_test.shape[0]
            indata_train, result_train = x_train, y_train
            indata_test, result_test = x_test, y_test
        elif dataset == "UCI_HAR":
            ntrain,indata_train,result_train = bp.read_inputs(trainfile,scale="gauss",scalereturn=0,verbose=False)
            ntest,indata_test,result_test = bp.read_inputs(testfile,scale="gauss",verbose=False)
        elif dataset == "mnist":
            ntrain,indata_train,result_train = bp.read_inputs(None,data=data_train,scale="gauss",scalereturn=0,verbose=False)
            ntest,indata_test,result_test = bp.read_inputs(None,data=data_test,scale="gauss",verbose=False)
        else:
            ntrain,indata_train,result_train,info = bp.read_inputs(trainfile,scale="gauss",scalereturn=1,verbose=False)
            ntest,indata_test,result_test = bp.read_inputs(testfile,scale="gauss",scaleparam=info,verbose=False)

        traindata = indata_train.copy()
        trainresult = result_train.copy()
        testdata = indata_test.copy()
        testresult = result_test.copy()

        # set the initial backpropogation weights
        if loadModelPath is None:
            bp.random_weights(scale=matscale,pos_init=pos_init,verbose=False)
        else:
            bp.read_weights_crossSim(loadPath=loadModelPath,verbose=False)

        # Print title of
        print("\niteration // avg loss (train) // max loss (train) // fraction correct (test)")    

        if verbose:
            print("\n")    
            # Print the title of the table so that it can be interpreted by plot tools
            # title is enclosed by '#' chars, parameters specified by keyword1=value1 keyword2=value2
            if model == "lookup":
                if params.algorithm_params.disable_clipping:
                    print ("#Training model="+model+" clip=disabled")
                else:
                    print("#Training model="+model)
                    if params.periodic_carry_params.use_periodic_carry:
                        print(" number_base="+str(params.periodic_carry_params.number_base))
                        print(" carry_threshold="+str(params.periodic_carry_params.carry_threshold))
                        print(" carry_frequency="+str(params.periodic_carry_params.carry_frequency))
                        print(" read_low_order_bits="+str(params.periodic_carry_params.read_low_order_bits))
                        print(" exact_carries="+str(params.periodic_carry_params.exact_carries))
                        print(" zero_reset="+str(params.periodic_carry_params.zero_reset))
                        print(" min_carry_update=" + str(params.periodic_carry_params.min_carry_update))
                        xbar_params1 = bp.ncores[0].neural_core.cores[1].params.xbar_params
                        if not perceptron:
                            xbar_params2 = bp.ncores[1].neural_core.cores[1].params.xbar_params
                    else:
                        xbar_params1 = bp.ncores[0].neural_core.params.xbar_params
                        if not perceptron:
                            xbar_params2 = bp.ncores[1].neural_core.params.xbar_params

                    min_update1, max_update1, min_update2, max_update2 = 0, 0, 0, 0
                    max_update1 = xbar_params1.row_update.maximum*xbar_params1.col_update.maximum/(xbar_params1.weights.range)*alpha
                    if xbar_params1.row_update.levels is not None:
                        min_update1 = xbar_params1.row_update.range*xbar_params1.col_update.range/(xbar_params1.weights.range)*alpha/(xbar_params1.row_update.levels-1)/(xbar_params1.col_update.levels-1)                
                    if not perceptron:
                        max_update2 = xbar_params2.row_update.maximum*xbar_params2.col_update.maximum/(xbar_params2.weights.range)*alpha
                        if xbar_params2.row_update.levels is not None:
                            min_update2 = xbar_params2.row_update.range*xbar_params2.col_update.range/(xbar_params2.weights.range)*alpha/(xbar_params2.row_update.levels-1)/(xbar_params2.col_update.levels-1)

                    print (" clip="+str(clipscale)+
                           "\n crossbar type = "+str(params.algorithm_params.crossbar_type.name)+
                           "\n row update bits="+str(params.xbar_params.row_update.bits)+" stochastic "+str(params.xbar_params.row_update.stochastic_rounding)+
                           "\n col update bits="+str(params.xbar_params.col_update.bits)+" stochastic "+str(params.xbar_params.col_update.stochastic_rounding)+
                           "\n row input bits="+str(params.xbar_params.row_input.bits)+" stochastic "+str(params.xbar_params.row_input.stochastic_rounding)+
                           "\n col input bits="+str(params.xbar_params.col_input.bits)+" stochastic "+str(params.xbar_params.col_input.stochastic_rounding)+
                           "\n row output bits="+str(params.xbar_params.row_output.bits)+" stochastic "+str(params.xbar_params.row_output.stochastic_rounding)+
                           "\n col output bits="+str(params.xbar_params.col_output.bits)+" stochastic "+str(params.xbar_params.col_output.stochastic_rounding)+
                           "\n row update max="+str(params.algorithm_params.row_update.maximum)+" col update max="+str(params.algorithm_params.col_update.maximum)+
                           "\n row input max="+str(params.algorithm_params.row_input.maximum)+" col input max="+str(params.algorithm_params.col_input.maximum)+
                           "\n row output max="+str(params.algorithm_params.row_output.maximum)+" col output max="+str(params.algorithm_params.col_output.maximum)+
                           "\n mat1 max update="+str(max_update1)+" mat2 max update="+str(max_update2)+
                           "\n mat1 min update=" + str(min_update1) + " mat2 min update=" + str(min_update2))
            else:
                print("#Training model="+model)
            print(" seed="+str(seed)+", activation function="+activate+", output activation function="+activate_output+", learning rate="+str(alpha)+
               " #\n")

        if useGPU:
            import cupy as cp
            cp.cuda.Device(0).use()
            testdata = cp.array(testdata)
            traindata = cp.array(traindata)

        bp.ndata = ntest
        bp.answers = testresult
        bp.indata = testdata

        count, frac = bp.classify(n=ncset)
        print("%d // %g // %g // %g" % (0, 0.0, 0.0, frac))

        # load the training data
        bp.ndata = ntrain
        bp.indata = traindata
        bp.answers = trainresult
        time1 = time.time()

        # loop over training, one iteration at a time
        # do accuracy test on test data at each iteration
        results = np.zeros(n_epochs)
        for k in range(n_epochs):
            if batchsize == 1:
                aveerror, maxerror = bp.train(n=ntset, i_epoch=k, debug=1)
            else:
                aveerror, maxerror = bp.train_batch_digital(n=ntset, batchsize=batchsize, i_epoch=k, debug=1)

            # load test data and classify
            bp.ndata = ntest
            bp.indata = testdata
            bp.answers = testresult
            count, frac = bp.classify(n=ncset, debug=1)

            # reload training data
            bp.ndata = ntrain
            bp.indata = traindata
            bp.answers = trainresult

            # print and store results
            # Accuracy/loss table: epoch / avg training loss / max training loss / test accuracy
            print("%d // %g // %g // %g" % (k + 1, aveerror, maxerror, frac))
            results[k] = frac

        if saveModelPath is not None:
            bp.write_weights_npz(savePath=saveModelPath)

        diagnostics = {'target_updates':None, 'real_updates':None}
        if params is not None and params.analytics_params.record_updates:
            diagnostics['target_updates'], diagnostics['real_updates'] = bp.ncores[0].get_update_record()
            if useGPU:
                diagnostics['target_updates'] = cp.asnumpy(diagnostics['target_updates'])
                diagnostics['real_updates'] = cp.asnumpy(diagnostics['real_updates'])


        # print timings
        cpu = time.time() - time1
        print("\nTraining time = %g" % cpu)
        cpu = time.time() - time_start
        print("Total time = %g\n" % cpu)

        if model == "numeric":
            print('Baseline mat parameters')
            for k in range(Nlayers):
                weights_k = bp.ncores[k].matrix
                if useGPU:
                    weights_k = cp.asnumpy(weights_k)
                print("Layer "+str(k+1)+" min: {:.4f}.".format(np.min(weights_k)) + ", max: {:.4f}.".format(np.max(weights_k)))

        ##### plot histogram of trained weights
        if plot_weight_hist:
            if model=="lookup":
                fig = plt.figure(figsize=(1.75,1.75))
                clip_min = bp.ncores[0].neural_core.params.xbar_params.weight_clipping.minimum
                clip_max = bp.ncores[0].neural_core.params.xbar_params.weight_clipping.maximum
                #scale matrix back into conductance range in uS
                scaling = bp.ncores[0].neural_core.params.numeric_params.dG_lookup.Gmax_clip/clip_max*1e6
                matrix = bp.ncores[0].neural_core.core.matrix * scaling
                n, bins, patches = plt.hist(matrix.flatten(), normed=1, bins=100, range=(clip_min* scaling,clip_max* scaling))
                plt.xlabel(r"Conductance ($\mu$S)")
                plt.ylabel("Probability Density")
                plt.xlim([clip_min* scaling,clip_max* scaling])
                print("the weight minimum is ",bp.ncores[0].neural_core.params.xbar_params.weights.minimum)
                print("the weight maximum is ",bp.ncores[0].neural_core.params.xbar_params.weights.maximum)
                print("the weight clipping minimum is ",clip_min)
                print("the weight clipping maximum is ",clip_max)

            elif model=="numeric":
                weights0 = bp.ncores[0].matrix
                weights1 = bp.ncores[1].matrix
                print("Layer 0: The min/max weight is ",weights0.min(),weights0.max() )
                print("Layer 1: The min/max weight is ",weights1.min(),weights1.max() )
                print("Layer 0: The standard deviation is  ",np.std(weights0))
                print("Layer 1: The standard deviation is ",np.std(weights1) )
                fig = plt.figure(figsize=(1.75,1.75))
                n, bins, patches = plt.hist(weights0.flatten(), normed=1, bins=50)
                plt.xlabel(r"Weight")
                plt.ylabel("Probability Density")
                plt.title("layer0")
                fig = plt.figure(figsize=(1.75,1.75))
                n, bins, patches = plt.hist(weights1.flatten(), normed=1, bins=50)
                plt.xlabel(r"Weight")
                plt.ylabel("Probability Density")
                plt.title("layer1")

            else:
                raise ValueError("Unknown model"+str(model))

        return results, diagnostics


    def extract_plotting_data(self,filenames):
        """
        extracts the error and epoch count for each file name passed

        :param filenames:
        :type filenames: list[str]
        :return: lists of numpy arrays: epoch and error
        """
        ########## load data using plot tools (a set of tools designed to interpret the files saved by train)

        import helpers.plot_tools as PT
        e = PT.Extract()
        titles,tables = e.table_read_all(filenames, {})
        epoch, error  = [], []
        for i,table in enumerate(tables):
          epoch.append( PT.extract_column_table(table,1) )
          accuracy = PT.extract_column_table(table,4)
          error.append( [100.0 - 100.0*value for value in accuracy] )

        return epoch,error


    def plot_training(self, filenames, dataset=None, plot_linear=False, legend = ("Exp. Derived","Ideal Numeric"), ylim =None, plot_small=False, plot_title="", save_filename = "accuracy_vs_epoch.png", xlim=None):
        """
        Creates and saves a plot of the training results

        :param filenames: list of filenames of the data to plot
        :param dataset: (deprecated, use plot_title and save_filename instead)  what dataset to use for labels:  small, large, cyber
        :param outdir: the output directory to store figures to
        :param plot_linear: Use a linear scale or a log scale on the plot
        :param plot_small:  make smaller figures suitable for IEDM/VLSI papers
        :param plot_title: What title to add to the plot if any
        :param save_filename: What filename to save the plot to, filename extension should be specifed to specify output file type
        :return:
        """

        epoch,error = self.extract_plotting_data(filenames)

        ###########  create plot

        if not plot_small:
            fig = plt.figure() #use default figsize
            ax = fig.add_subplot(111)
            plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
        else:
            # plot settings of VLSI/IEDM sized figs
            fig = plt.figure(figsize=(1.1,1.1))
            ax = fig.add_subplot(111)
            mpl.rcParams['font.size'] = 6
            plt.xticks([0,10,20,30,40,50,60,70,80,90,100])

        for ind in range(len(epoch)):
            if plot_linear:
                plt.plot(epoch[ind],100-np.array(error[ind]),'-', linewidth=1)
            else:
                plt.semilogy(epoch[ind],error[ind],'-', linewidth=1)#, color=colors[ind])

        if not plot_linear:
            ax.invert_yaxis()
            ax.set_yticklabels(['100','99','90','0'])

        mpl.rcParams['axes.linewidth']=0.5
        mpl.rcParams['axes.titlesize']='medium'
        if plot_linear:
            fig.subplots_adjust(left=0.22, bottom=0.17)
            plt.ylim([0,100])
        else:
            if not plot_small:
                fig.subplots_adjust(left=0.185, bottom=0.175, top=0.9, right = 0.95)
            else:
                fig.subplots_adjust(left=0.22, bottom=0.21, top=0.91, right = 0.95)
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)
        plt.legend(legend,loc=4,frameon=False, fontsize=7)
        plt.xlabel("Training Epoch", labelpad=0.5)
        plt.ylabel("Accuracy", labelpad=0.5)

        # maintain support for deprecated option "dataset"
        if dataset =="small":
            plot_title = "Small Digits"
            save_filename = "small_image.png"
        elif dataset == "mnist":
            plot_title = "MNIST"
            save_filename = "large_image.png"
        elif dataset == "cyber":
            plot_title = "File Types"
            save_filename = "file_types.png"
        elif dataset == "iris":
            plot_title = "Iris Classification"
            save_filename = "iris.png"
        elif dataset == "SPECTF":
            plot_title = "SPECTF"
            save_filename = "SPECTF.png"
        elif dataset == "fashion":
            plot_title = "fashion"
            save_filename = "fashion.png"

        plt.title(plot_title) # 64x36x10
        plt.savefig(os.path.join(self.outdir,save_filename),dpi=1200)

    def plot_avg_std(self,x_vec,averages,stdevs,dataset,x_type="epochs"):

        if dataset =="small":
            plot_title = "Small Digits: LUT Variability"
            save_filename = "small_image_var.png"
        elif dataset == "mnist":
            plot_title = "MNIST: LUT Variability"
            save_filename = "large_image_var.png"
        elif dataset == "cyber":
            plot_title = "File Types: LUT Variability"
            save_filename = "file_types_var.png"
        elif dataset == "iris":
            plot_title = "Iris Classification: LUT Variability"
            save_filename = "iris_var.png"
        elif dataset == "SPECTF":
            plot_title = "SPECTF: LUT Variability"
            save_filename = "SPECTF_var.png"
        elif dataset == "fashion":
            plot_title = "fashion"
            save_filename = "fashion.png"

        def errorfill(x, y, yerr, style,label,color=None, alpha_fill=0.05, ax=None):
            ax = ax if ax is not None else plt.gca()
            if color is None:
                color = ax._get_lines.color_cycle.next()
            if np.isscalar(yerr) or len(yerr) == len(y):
                ymin, ymax = y - yerr/2, y + y_err/2
            elif len(yerr) == 2:
                ymin, ymax = yerr
            ax.plot(x, y, style, label =label, color=color)
            ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

        colors = ['r','g','violet','b','k']
        linsty = ['-.o','-+','-^','-o','-*',]
        dict = ['Numeric','Standard LUT','Variable LUT: Good,Small','Variable LUT: Good','Variable LUT: Wafer-Scale']
        [errorfill(x_vec,averages[:,x],  stdevs[:,x], linsty[x],label =dict[x], color = colors[x])
        for x in range(averages.shape[1])]
        plt.legend(bbox_to_anchor=(1.04,1), loc="bottom left", ncol = 1,fontsize = 10)
        plt.ylabel('% Correct Test Set',fontsize = 13)
        if x_type == "epochs":
            plt.xlabel('Training Epoch')
        elif x_type == "lr":
            plt.xlabel('Learning Rate')
        plt.title(plot_title)
        print(self.outdir)
        plt.savefig(os.path.join(self.outdir,save_filename),dpi=1200,bbox_inches='tight')

    def plot_avg_std_lr(self, lr_vec,averages,stdevs,dataset):
        self.plot_avg_std(lr_vec,averages,stdevs,dataset,x_type="lr")