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

# Use this file to produce a list of ADC (min, max) ranges for each layer of ResNet50
# This is to be used for files that do not use either input or weight bit slicing
# This allows an optimizer function to be used to set the limits because there are no real hardware constraints on the limits

# Folder where profiled values are stored
profiling_folder = "./adc_dac_inputs/ADCinputs_1152rows_1slices_BALANCED_ibits/"
# npy output file path
# output_name = "./adc_limits/examples/ADClimits_1152rows_1slices_OFFSET_ibits.npy"
output_name = "./adc_limits/imagenet/ADClimits_1152rows_1slices_BALANCED_ibits.npy"

# Whether profile_ADC_biased was True for the ADC inputs
reluAwareLimits = False

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

###
### ResNet50 specific parameters
###

# Layer nums for ResNet conv and dense layers
layerNums = [0,2,3,4,5,7,8,9,11,12,13,15,16,17,18,20,21,22,24,25,26,28,29,30,\
	32,33,34,35,37,38,39,41,42,43,45,46,47,49,50,51,53,54,55,57,58,59,60,62,63,64,66,67,68,71]

###
### Quantization loss functions
###

# Optimize both the min and the max
# Use for ADC range of split cores and any layers not immediately preceding a ReLU
def quantizationError_minMax(etas,x,Nbits,norm_ord):
	# Clip
	etaMin, etaMax = etas
	P_min = 100*pow(10,etaMin)
	P_max = 100*(1 - pow(10,etaMax))
	P_min = np.clip(P_min,0,100)
	P_max = np.clip(P_max,0,100)

	ADCmin = ncp.percentile(x,P_min)
	ADCmax = ncp.percentile(x,P_max)
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
##  Determine ADC ranges
##
################

ADC_limits = np.zeros((len(layerNums),2))

for k in range(len(layerNums)):
	print('Layer '+str(k)+' ('+str(layerNums[k])+')')
	
	Nbits_in = (7 if k == 0 else 8)

	x_dist_ADC = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+"_ibit0.npy")
	for j in range(1,Nbits_in):
		x_dist_ADC_j = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+"_ibit"+str(j)+".npy")
		x_dist_ADC = cp.concatenate((x_dist_ADC,x_dist_ADC_j))

	ADC_max = ncp.asnumpy(ncp.max(x_dist_ADC))
	ADC_min = ncp.asnumpy(ncp.min(x_dist_ADC))

	# Optimize the ADC percentile
	if Nbits > 0:
		etas0 = (-4, -4)
		eta = minimize(quantizationError_minMax,etas0,args=(x_dist_ADC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
		Pmin_ADC = 100*pow(10,eta.x[0])
		Pmax_ADC = 100*(1-pow(10,eta.x[1]))
		print('    ADC Percentiles: {:.3f}'.format(Pmin_ADC)+', {:.3f}'.format(Pmax_ADC))
		xmin_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmin_ADC))
		xmax_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmax_ADC))
	else:
		xmin_ADC = ADC_min
		xmax_ADC = ADC_max
	clipped = ((ADC_max-xmax_ADC) + (xmin_ADC-ADC_min))/(ADC_max-ADC_min)
	print('    Percentage of profiled ADC range clipped: {:.3f}'.format(clipped*100)+'%')

	# Set the limits
	ADC_limits[k,:] = np.array([xmin_ADC,xmax_ADC])

# Save the generated limits to a npy file
np.save(output_name,ADC_limits)