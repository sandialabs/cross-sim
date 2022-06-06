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

# Use this file to determine ADC limits for systems with weight bit slicing
# Express ADC limits instead of power of 2 by which uncalibrated range is clipped
# This constraint makes it simple to combine bit slices, even if they have different ADC limits
Nslices = 8
NrowsMax = 1152
Wbits_slice = int(8 / Nslices)
xbar_style = "BALANCED"

# Folder where profiled values are stored
# profiling_folder = "./adc_dac_inputs/ADCinputs_1152rows_4slices_OFFSET_ibits/"
profiling_folder = "./adc_dac_inputs/ADCinputs_1152rows_8slices_BALANCED/"

# npy output file path
# output_name = "./adc_limits/examples/ADClimits_1152rows_4slices_OFFSET_ibits.npy"
output_name = "./adc_limits/imagenet_bitslicing/ADClimits_1152rows_8slices_BALANCED_99.9995.npy"

# Calibrated limits must include this percentile on both sides
# e.g. 99.99 means 99.99% and 0.01% percentile will be included in the range
# Warning: for some layers, a vast majority of ADC inputs may be zero and pct may have to be raised to
# get a meaningful range
pct = 99.9995

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

NrowsList_unsplit = np.array([147,64,576,64,64,256,576,64,256,576,64,256,1152,128,256,512,1152,128,512,1152,128,512,1152,128,\
	512,2304,256,512,1024,2304,256,1024,2304,256,1024,2304,256,1024,2304,256,1024,2304,256,1024,4608,512,\
	1024,2048,4608,512,2048,4608,512,2048])

# Get the actual number of rows
NrowsList = np.zeros(len(layerNums),dtype=int)
for k in range(len(layerNums)):
	if NrowsList_unsplit[k] <= NrowsMax:
		NrowsList[k] = NrowsList_unsplit[k]
	else:
		Ncores = (NrowsList_unsplit[k]-1)//NrowsMax + 1
		if NrowsList_unsplit[k] % Ncores == 0:
			NrowsList[k] = (NrowsList_unsplit[k] // Ncores).astype(int)
		else:
			Nrows1 = np.round(NrowsList_unsplit[k] / Ncores)
			Nrows2 = NrowsList_unsplit[k] - (Ncores-1)*Nrows1
			if Nrows1 != Nrows2:
				NrowsList[k] = np.maximum(Nrows1,Nrows2).astype(int)

################
##
##  Determine ADC ranges
##
################

clip_power = np.zeros((len(layerNums),Nslices))

# To display some clipping statistics only (not needed to get limits)
percentile_factors = np.zeros((len(layerNums),Nslices))

print('# rows: '+str(NrowsMax))
print('Percentile: {:.4f}'.format(pct))

for k in range(len(layerNums)):

	# Bring # rows to nearest power of 2
	ymax = pow(2,np.round(np.log2(NrowsList[k])))
	# Correct to make level separation a multiple of the min cell current
	ymax *= pow(2,Wbits_slice)/(pow(2,Wbits_slice)-1)
	Nbits_in = 8
	if k == 0:
		ymax *= pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
	else:
		ymax *= pow(2,Nbits_in)/(pow(2,Nbits_in)-1)

	for i_slice in range(Nslices):
		x_dist_ADC_i = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+"_slice"+str(i_slice)+".npy")
		x_dist_ADC_i /= ymax

		pct_k = pct

		# ADC inputs can be signed: always true for balanced, also true for first layer (which has signed inputs)
		if k == 0 or xbar_style == "BALANCED":
			p_neg = ncp.percentile(x_dist_ADC_i,100-pct_k)
			p_pos = ncp.percentile(x_dist_ADC_i,pct_k)
			p_out = np.maximum(np.abs(p_neg),np.abs(p_pos))
			clip_power_k = np.floor(np.log2(1/p_out)).astype(int)
		else:
			p_out = ncp.percentile(x_dist_ADC_i,pct_k)
			clip_power_k = np.floor(np.log2(1/p_out)).astype(int)

		if clip_power_k > 25 and pct < 99.999:
			print("Redo with higher pct")
			pct_k = 99.999
			if k == 0 or xbar_style == "BALANCED":
				p_neg = ncp.percentile(x_dist_ADC_i,100-pct_k)
				p_pos = ncp.percentile(x_dist_ADC_i,pct_k)
				p_out = np.maximum(np.abs(p_neg),np.abs(p_pos))
				clip_power_k = np.floor(np.log2(1/p_out)).astype(int)
			else:
				p_out = ncp.percentile(x_dist_ADC_i,pct_k)
				clip_power_k = np.floor(np.log2(1/p_out)).astype(int)

		print("Layer "+str(layerNums[k])+" ("+str(NrowsList[k])+" rows), slice "+str(i_slice)+" clip power: "+str(clip_power_k)+" bits")
		clip_power[k,i_slice] = clip_power_k

# Save the generated limits to a npy file
np.save(output_name,clip_power)