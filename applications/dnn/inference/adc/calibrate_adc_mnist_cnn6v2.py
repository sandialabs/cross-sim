#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


import numpy as np
from scipy.optimize import minimize
import time
import os

# Use this file to produce a list of ADC (min, max) ranges for each layer of CNN6_v2 on MNIST
# This is to be used for files that do not use either input or weight bit slicing

# Max number of rows
NrowsMax = 1152

# Mapping style
xbar_style = "BALANCED"

# Whether profile_ADC_reluAware was True for the ADC inputs
reluAwareLimits = True

# Input bit slicing (must be False if reluAwareLmits = True)
input_bitslicing = False

##### Folder where profiled values are stored
##### Make sure the correct file is selected that is consistent with the options above!
ibit_msg = ("_ibits" if input_bitslicing else "")
relu_msg = ("_reluAware" if reluAwareLimits else "")

profiling_folder = "./profiled_adc_inputs/mnist_CNN6_v2_"+\
			str(NrowsMax)+"rows_1slices_"+xbar_style+ibit_msg+relu_msg+"/"

##### npy output file path
output_name = "./adc_limits/examples/adc_limits_CNN6v2_"+\
				str(NrowsMax)+"rows_1slices_"+xbar_style+ibit_msg+relu_msg+".npy"

# Quantization loss function settings
Nbits = 12 # not necessarily the ADC resolution to be used
norm_ord = 1

# Enable GPU
useGPU = True

if useGPU:
	import cupy as cp
	gpu_num = 0
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
	ncp = cp
else:
	ncp = np

###
### ResNet50 specific parameters
###

# Layer nums for ResNet conv and dense layers
layerNums = [0,1,3,4,6,7]

# Number of matrix partitions for each layer, the list below assumes NrowsMax = 1152 has been used
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
ncoresList = [1,1,1,1,1,1]

# Which layers use ReLU (1) and which do not (0)
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
reluList =   [1,1,1,1,1,0]

###
### Quantization loss functions
###

# Optimize the max only
# Use for the ADC range of non-split cores preceding a ReLU
# Use for all DAC ranges
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
	return float(err)

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
	return float(err)

################
##
##  Determine ADC ranges
##
################

ADC_limits = np.zeros((len(layerNums),2))

for k in range(len(layerNums)):
	print('Layer '+str(k)+' ('+str(layerNums[k])+')')
	
	x_dist_ADC = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+".npy")
	x_dist_ADC = x_dist_ADC.flatten()

	ADC_max = ncp.asnumpy(ncp.max(x_dist_ADC))
	ADC_min = ncp.asnumpy(ncp.min(x_dist_ADC))

	# Optimize the ADC percentile
	# If ADC inputs are ReLU-aware, layer is not split, and layer has a ReLU, optimize only the max
	# The min will be set to the minimum value in the profiled set which will correspond to a post-ReLU value of zero for the
	# output channel with the most positive bias value
	if reluAwareLimits and not input_bitslicing and ncoresList[k] == 1 and reluList[k] == 1:
		xmin_ADC = ADC_min
		if Nbits > 0:
			eta0 = -4
			eta = minimize(quantizationError_max,eta0,args=(x_dist_ADC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
			percentile_ADC = 100*(1-pow(10,eta.x[0]))
			xmax_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,percentile_ADC))
		else:
			xmax_ADC = ADC_max
		clipped = ((ADC_max-xmax_ADC) + (xmin_ADC-ADC_min))/(ADC_max-ADC_min)
		print('    Percentage of ADC range clipped: {:.3f}'.format(clipped*100)+'%')

	else:
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

print("Calibrated ADC limits:")
print(ADC_limits)

# Save the generated limits to a npy file
np.save(output_name,ADC_limits)