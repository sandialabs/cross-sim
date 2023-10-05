#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import time
import os

# Use this file to produce a list of ADC (min, max) ranges for each layer of VGG-19
# This is to be used for files that do not use either input or weight bit slicing
# This specific file sets the limits based on a fixed percentile range of the profiled ADC input distribution

# Percentile of the ADC input distribution used to set ADC limits
percentile = 99.370

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

profiling_folder = "./profiled_adc_inputs/imagenet_VGG19_"+\
			str(NrowsMax)+"rows_1slices_"+xbar_style+ibit_msg+relu_msg+"/"

##### npy output file path
output_name = "./adc_limits/examples/adc_limits_VGG19_"+\
				str(NrowsMax)+"rows_1slices_"+xbar_style+ibit_msg+relu_msg+".npy"

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
### VGG-19 specific parameters
###

# Layer nums for ResNet conv and dense layers
layerNums = [0,1,3,4,6,7,8,9,11,12,13,14,16,17,18,19,22,23,24]

# Number of matrix partitions for each layer, the list below assumes NrowsMax = 1152 has been used
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
ncoresList = [1,1,1,1,1,2,2,2,2,4,4,4,4,4,4,4,22,4,4]

# Which layers use ReLU (1) and which do not (0)
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
reluList = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]

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

	# Set the ADC limits based on fixed percentile
	# If ADC inputs are ReLU-aware, layer is not split, and layer has a ReLU, optimize only the max
	# The min will be set to the minimum value in the profiled set which will correspond to a post-ReLU value of zero for the
	# output channel with the most positive bias value
	if reluAwareLimits and ncoresList[k] == 1 and reluList[k] == 1:
		xmin_ADC = ADC_min
		if percentile < 100:
			xmax_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,percentile))
		else:
			xmax_ADC = ADC_max
		clipped = ((ADC_max-xmax_ADC) + (xmin_ADC-ADC_min))/(ADC_max-ADC_min)
		print('    Percentage of ADC range clipped: {:.3f}'.format(clipped*100)+'%')

	else:
		if percentile < 100:
			Pmin_ADC = (100 - percentile)/2
			Pmax_ADC = percentile + (100 - percentile)/2
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