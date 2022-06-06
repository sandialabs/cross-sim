#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from scipy.optimize import minimize
import time
import os

# Use this file to produce a list of DAC (min, max) ranges for each layer of ResNet50
# DAC limits should be independent of most hardware settings such as input bitslicing, 
# weight bitslicing, # rows, and negative number handling, but are affected by fold_batchnorm, 
# weight_bits, etc

# Folder where profiled values are stored
profiling_folder = "./adc_dac_inputs/VGG19_DACinputs_1152rows_1slices_BALANCED/"
# npy output file path
output_name = "./adc_limits/examples/VGG19_DAClimits_1152rows_1slices_BALANCED.npy"

# Quantization loss function settings
Nbits = 12 # not necessarily the ADC resolution to be used
norm_ord = 1

# Enable GPU
useGPU = True

if useGPU:
	import cupy as cp
	gpu_num = 1
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
	ncp = cp
else:
	ncp = np

# Layer nums for ResNet conv and dense layers
layerNums = [0,1,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23]

# Quantization loss function
def quantizationError_max(eta,x,Nbits,norm_ord):
	# Clip
	P = 100*(1 - pow(10,eta))
	P = np.clip(P,0,100)

	ADCmin = ncp.min(x)
	ADCmax = ncp.percentile(x,P)
	x_Q = x.copy()
	x_Q = x_Q.clip(ADCmin,ADCmax)

	# Quantize
	qmult = (2**Nbits-1) / (ADCmax - ADCmin)
	x_Q = (x_Q - ADCmin)*qmult
	x_Q = ncp.rint(x_Q,out=x_Q)
	x_Q /= qmult
	x_Q += ADCmin

	err = ncp.linalg.norm(x-x_Q,ord=norm_ord)
	return err

################
##
##  Determine DAC ranges
##
################
DAC_limits = np.zeros((len(layerNums),2))

# First layer's DAC limits come directly from the limits of the preprocessed ImageNet test set
# Using these limits for the first layer ensures no clipping will occur
DAC_limits[0,:] = np.array([-124,152])

for k in range(1,len(layerNums)):
	print('Layer '+str(k))	
	x_dist_DAC = ncp.load(profiling_folder+"dac_inputs_layer"+str(layerNums[k])+".npy")

	DAC_max = ncp.asnumpy(ncp.max(x_dist_DAC))
	DAC_min = ncp.asnumpy(ncp.min(x_dist_DAC))

	# Optimize the DAC percentile
	if Nbits > 0:
		eta0 = -4
		eta = minimize(quantizationError_max,eta0,args=(x_dist_DAC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
		percentile_DAC = 100*(1-pow(10,eta.x[0]))
		print('    DAC Percentile: {:.3f}'.format(percentile_DAC))
		xmin_DAC = ncp.asnumpy(ncp.min(x_dist_DAC))
		xmax_DAC = ncp.asnumpy(ncp.percentile(x_dist_DAC,percentile_DAC))
	else:			
		xmin_DAC = DAC_min
		xmax_DAC = DAC_max
	clipped = (DAC_max-xmax_DAC)/(DAC_max-DAC_min)
	print('    Percentage of DAC range clipped: {:.3f}'.format(clipped*100)+'%')

	# Set the limits
	DAC_limits[k,:] = np.array([0,xmax_DAC])

# Save the generated limits to a npy file
np.save(output_name,DAC_limits)