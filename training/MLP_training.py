#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import sys, os
from training_net import train_neural_net
from lr_schedule import adjust_learning_rate
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)

'''
Analog in-memory training setup script (multi-layer perceptron)
========

This file contains the configuration for a set of training simulations
Once parameters are set, run this file directly: python MLP_training.py

By default, this script runs:
	- one set of runs with ideal numeric weight updates (ideal devices)
	- one set of runs using the set specified by "lookup_table_single", which
		contains the LUT of a single device
	- one set of runs using the set specified by "lookup_table_multi", which
		contains LUTs measured on multiple devices. Array elements are randomly
		assigned to these LUTs at the beginning of training to model the effects
		of device-to-device variations in the update LUT

This script can be adapted to run any sequence of simulations desired, or to sweep
parameters (e.g. learning rate)

Before running this script, make sure the following empty sub-directories exist in the
training directory:
- console_output
- cross_sim_models
- sweep_results
- weight_update_stats
'''

# GPU settings
useGPU = True
if useGPU:
	gpu_num = 0 # ID of GPU to use
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

# Directory to save console output
outdir = "console_output/"

### Sweep parameters
# Does not include learning rate sweep: see MLP_training_lr
saveSweep = True # save accuracy table in a csv
Nruns = 1 # number of models to run with identical settings
Nepochs = 20 # number of epochs per model

### Which lookup table(s) to run
device_type = "DWMTJ"

if device_type == "TaOx":
	## TaOx ReRAM devices fabricated at Sandia, ca. 2016
	lookup_table_single = "TaOx"
	# Three options corresponding to different groups of TaOx devices
	# These are progressively filtered based on LUT quality
	# See: https://ieeexplore.ieee.org/abstract/document/8720596/
	# lookup_table_multi = "TaOx_large_set"
	lookup_table_multi = "TaOx_medium_set"
	# lookup_table_multi = "TaOx_small_set"

elif device_type == "ENODe":
	## Electrochemical RAM devices, Sandia and Stanford
	# See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8060477/pdf/fnins-15-636127.pdf
	# Multi lookup table based on 9 devices
	useGPU = False # Uses non-uniform look up table grid
	lookup_table_single = "ENODe"
	lookup_table_multi = "ENODe_multi"

elif device_type == "DWMTJ":
	## Domain wall magnetic tunnel junction synapses, Sandia and UT Austin
	# Based on micromagnetic simulation results using MuMax3
	mechanism = "STT" # mechanism of DW motion, "STT" or "SOT"
	temperature = 300 # operating temperature in K, 300 or 400
	lookup_table_single = "DWMTJ_"+mechanism+"_"+str(temperature)+"K"
	lookup_table_multi = "DWMTJ_"+mechanism+"_"+str(temperature)+"K_multi"

# Batch size
# If batch size = 1, devices are updated via a parallel outer product update
# If batch size > 1, updates are accumulated at floating-point precision until applied to devices at end of batch
# 	Devices are assumed to be programmed serially (not restricted to rank 1)
batchsize = 10

### Dataset
# task = "iris"
#task = "cyber"
#task = "SPECTF"
# task = "small"
# task = "mnist"
task = "fashion"
# task = "UCI_HAR"

# MLP network topology
# List of feature map dimensions (x, y, channel), starting from first layer input to last layer output
# Only MLPs are currently supported, so first (x,y) are always (1,1)
# Do not include the bias unit input
# If sizes not supplied to train(), a default size will be chosen (see training_net.py)
if task == "iris":
	sizes = (4,8,3)
elif task == "cyber":
	sizes = (256,150,9)
elif task == "SPECTF":
	sizes = (44,22,2)
elif task == "small":
	sizes = (64,36,10)
elif task == "mnist":
	sizes = (784,300,10)
elif task == "fashion":
	sizes = (784,300,10)
elif task == "UCI_HAR":
	sizes = (561,200,6)

# Dataset truncation: set the desired # examples for training set (ntset) and test set (ncset)
# Zero for no truncation
ntset = 0
ncset = 0

# Load pre-trained CrossSim models from npz archive? If False, use random initial weights
# NOTE: as provided, the cross_sim_models directory is empty
loadCrossSimModels = False
# Save trained CrossSim models to npz archive?
saveCrossSimModels = True

# If either of the above is True, specify the load and save filepaths below
# Separate filepath for every trained model in the sweep
types = ["numeric","lookup_standard","lookup_multi"]
loadModelPaths = [[None for j in range(3)] for i_run in range(Nruns)]
saveModelPaths = [[None for j in range(3)] for i_run in range(Nruns)]
if saveCrossSimModels and not os.path.isdir('./cross_sim_models/'):
    os.makedirs('./cross_sim_models')
for j in range(3):
	for i_run in range(Nruns):
		if loadCrossSimModels:
			loadModelPaths[i_run][j] = "./cross_sim_models/MLP_"+task+"_"+types[j]+"_run"+str(i_run)+".npz"
		if saveCrossSimModels:
			saveModelPaths[i_run][j] = "./cross_sim_models/MLP_"+task+"_"+types[j]+"_run"+str(i_run)+".npz"

# Weight sign encoding: balanced core ("BALANCED") vs offset core ("OFFSET")
wtmodel = "BALANCED"

# Periodic carry settings
# For details see: https://ieeexplore.ieee.org/document/7998164
periodic_carry = False # whether to use periodic carry
pc_Nslices = 2 # number of bit slices, e.g. devices per weight to encode magnitude
pc_number_base = 16 # multiplicative factor between period carry slices

# Learn bias weights for every layer
learnbias = (True, True)

### Activation and A/D settings
# Activation used for every layer but the last
activate = "SIGMOID"
# Activation used for last layer
activate_output = "SOFTMAX"
# A/D and D/A model
a2dmodel="NONE"
# Use stochastic rounding in update
stochastic_updates = False

# Learning rate (alpha)
# LR schedule is used if enabled. The function in lr_schedule.py is used
lr_sched = True

# NOTE: The default learning rates below are not necessarily optimal for the chosen
# 	dataset and topology!
if task == "iris":
	alpha_numeric = 0.1
	alpha_lut_standard = 0.02
	alpha_lut_multi = 0.02
elif task == "SPECF":
	alpha_numeric = 0.1
	alpha_lut_standard = 0.1
	alpha_lut_standard = 0.1
elif task == "small":
	alpha_numeric = 0.05
	alpha_lut_standard = 0.006
	alpha_lut_multi = 0.012
elif task == "cyber":
	alpha_numeric = 0.00005
	alpha_lut_standard = 0.000025
	alpha_lut_multi = 0.000025
elif task == "mnist":
	alpha_numeric = 0.01
	alpha_lut_standard = 0.0025
	alpha_lut_multi = 0.001
elif task == "fashion":
	alpha_numeric = 0.0002
	alpha_lut_standard = 0.0002
	alpha_lut_multi = 0.0002
elif task == "UCI_HAR":
	alpha_numeric = 0.01
	alpha_lut_standard = 0.01
	alpha_lut_multi = 0.01
else:
	raise ValueError("Invalid task")

# Whether to collect weight update information
collect_weight_updates = True
if collect_weight_updates:
	# Number of updates to collect statistics over (first layer only)
	# Make sure this doesn't exceed the total number of device updates during training
	# (max # updates = # weights in 1st layer x # training examples x # epochs
	Nupdates_total = 100000
	diagnosticParams = [True,Nupdates_total]
else:
	diagnosticParams = [False,100]

# Results container
results = np.zeros((Nepochs,3,Nruns))
deltaW_info = [[None for i in range(3)] for j in range(Nruns)]
epoch_vec = np.arange(Nepochs)

# Begin training loop
print('Training on dataset: '+task+'\n')

for i_run in range(Nruns):

	train_net = train_neural_net(outdir)

	# Generate params objects: one core only, assume second core is identical

	params_numeric = train_net.set_params(task=task,lookup_table=None,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU)

	params_standard = train_net.set_params(task=task,lookup_table=lookup_table_single,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU,periodic_carry=periodic_carry,pc_number_base=pc_number_base,pc_Nslices=pc_Nslices)

	params_multi = train_net.set_params(task=task,lookup_table=lookup_table_multi,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU,periodic_carry=periodic_carry,pc_number_base=pc_number_base,pc_Nslices=pc_Nslices)

	# Train neural net

	print('##########')
	print('Numeric (ideal), Run '+str(i_run))
	print('##########')
	results[:,0,i_run], deltaW_info[i_run][0] = train_net.train(filename=task+"_numeric.txt",dataset=task,params=params_numeric,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_numeric,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][0],saveModelPath=saveModelPaths[i_run][0],sizes=sizes)

	print('##########')
	print('Single LUT, Run '+str(i_run))
	print('##########')
	results[:,1,i_run], deltaW_info[i_run][1] = train_net.train(filename=task+"_lookup_standard.txt",dataset=task,params=params_standard,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_lut_standard,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,\
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][1],saveModelPath=saveModelPaths[i_run][1],sizes=sizes)

	print('##########')
	print('Multi LUT, Run '+str(i_run))
	print('##########')
	results[:,2,i_run], deltaW_info[i_run][2] = train_net.train(filename=task+"_lookup_multi.txt",dataset=task,params=params_multi,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_lut_multi,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,\
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][2],saveModelPath=saveModelPaths[i_run][2],sizes=sizes)
	print('Done training '+task+', Multi LUT')

# Average results over identical runs
results_mean = np.mean(results,2)
results_std = np.std(results,2)

# Save results
if saveSweep:
	result_folder = "./sweep_results/"+task+"/"+device_type+"/"
	if result_folder is not None and not os.path.isdir(result_folder):
		os.makedirs(result_folder)
	np.savetxt(result_folder+wtmodel+"_noise_mean_all.csv",results_mean,delimiter=",")
	np.savetxt(result_folder+wtmodel+"_noise_std_all.csv",results_std,delimiter=",")

# Analyze the collected weight update statistics
if collect_weight_updates and not periodic_carry:
	diag_folder = "./weight_update_stats/"+task+"/"
	if diag_folder is not None and not os.path.isdir(diag_folder):
		os.makedirs(diag_folder)
	fout = diag_folder + "statistics_"+device_type+"_"+wtmodel+"writenoise.txt"
	fileW = open(fout,"w")
	for k in range(3):
		if k == 0:
			lut = 'numeric'
		elif k == 1:
			lut = lookup_table_single
		elif k == 2:
			lut = lookup_table_multi
		diagnostics_k = deltaW_info[0][k]
		target_updates = diagnostics_k['target_updates']
		real_updates = diagnostics_k['real_updates']

		N_updates = len(target_updates) # number of updates collected for statistical analysis
		target_updates /= (np.std(target_updates)*2)
		real_updates /= (np.std(real_updates)*2)
		update_error = target_updates - real_updates
		mean_err, std_err = np.mean(update_error), np.std(update_error)
		std_err_pos = np.std(update_error[target_updates>0])
		std_err_neg = np.std(update_error[target_updates<0])
		asym = abs(std_err_pos-std_err_neg)

		fileW.write('\nTask: '+str(task))
		fileW.write('\nLook-up table set: '+str(lut))
		fileW.write('\n# updates: '+str(N_updates))
		fileW.write("\nMean error: "+str(mean_err))
		fileW.write("\nSpread error: "+str(std_err))
		fileW.write("\nSpread positive error: "+str(std_err_pos))
		fileW.write("\nSpread negative error: "+str(std_err_neg))
		fileW.write("\nAsymmetry: "+str(asym))
		fileW.write("\n \n")

		fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
		plt.subplots_adjust(wspace=0.4)
		ax1.scatter(target_updates,real_updates,s=10)
		C = np.maximum(np.max(np.abs(target_updates)),np.max(np.abs(real_updates)))
		x = np.linspace(-C,C,1000)
		ax1.plot(x,x,'--k',linewidth=2)
		ax1.tick_params(labelsize=14)
		ax1.set_xlabel("Target update",fontsize=14)
		ax1.set_ylabel("Real update",fontsize=14)
		ax1.set_xlim(-C,C); ax1.set_ylim(-C,C)
		
		ax2.hist(update_error,bins=100)
		ax2.set_xlabel('Update error',fontsize=14) 
		ax2.set_ylabel('Probability density',fontsize=14)
		ax2.tick_params(labelsize=14)
		D = np.max(np.abs(update_error))
		if D > 0:
			ax2.set_xlim(-D,D)
		else:
			ax2.set_xlim(-1,1)
		save_filename = diag_folder + "innercore_update_error_"+wtmodel+"_"+lut+".png"
		fig.savefig(save_filename,dpi=600,bbox_inches='tight')

	fileW.close()