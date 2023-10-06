#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys
import numpy as np
sys.path.append("../..")
from simulator.cores.analog_core import AnalogCore
from simulator.algorithms.dsp.dft import DFT
from applications.mvm_params import set_params
from scipy.linalg import dft
from PIL import Image
import matplotlib.pyplot as plt

##########################################
## Settings
# There are three equivalent code implementations of the 2D analog DFT
# Use implemetation = 0, 1, 2 to switch between these implementations
implementation = 2
useGPU = True

if useGPU:
	import cupy as cp
	xp = cp
else:
	xp = np

xp.random.seed(95)

##########################################
## Inputs in this example consist of three 256x256 RGB images
N_ft = 256
N_ch = 3

# Select the image
image_id = "peppers"
# image_id = "orchid"
# image_id = "peacock"

X_image = Image.open('./example_images/'+image_id+'.jpg')
X_image = xp.array(X_image.resize((N_ft,N_ft))).astype(xp.float32)
X_image /= 255

###########################################
## Parameters
params = set_params(
	complex_input = True,
	complex_matrix = True,
	wtmodel = "BALANCED",
	error_model = "generic",
	alpha_error = 0.00,
	proportional_error = False,
	noise_model = "none",
	alpha_noise = 0.00,
	proportional_noise = True,
	t_drift = 0,
	drift_model = None,
	Rmin = 1e3,
	Rmax = 1e6,
	infinite_on_off_ratio = True,
	Rp_row = 0,
	Rp_col = 0,
	NrowsMax = 64,
	NcolsMax = 64,
	weight_bits = 8,
	adc_bits = 0,
	adc_range = (-1,1),
	dac_bits = 0,
	dac_range = (-1,1),
	interleaved_posneg = False,
	gate_input = False,
	Nslices = 1,	
	digital_offset = False,
	adc_range_option = "CALIBRATED",
	Icol_max = 1e6,
	balanced_style = "ONE_SIDED",
	input_bitslicing = False,
	ADC_per_ibit = False,
	useGPU = useGPU,
	gpu_id = 0,)


##########################################
## Generate DFT matrices in numpy
W_dft = dft(N_ft)
W_idft = np.matrix.getH(W_dft)

## Create cores (for implementations 0 and 1)
xbar_dft = AnalogCore(W_dft, params=params)
xbar_idft = AnalogCore(W_idft, params=params)

## Create DFT object (for implementation 2)
dftCore_fwd = DFT(N_ft, params=params)
dftCore_inv = DFT(N_ft, params=params, inverse=True)

##########################################
## Process 2D image reconstruction 
## (x -> DFT -> IDFT -> x)
X_recon = xp.zeros(X_image.shape, dtype=np.complex128)

###### MVM based implementation
if implementation == 0:

	# Container for inetermediate results
	Y = xp.zeros(X_image.shape,dtype=np.complex128)

	# Forward 2D DFT
	for ch in range(N_ch):
		y_imed = xp.zeros((X_image.shape[1],X_image.shape[0]),dtype=np.complex128)
		for k in range(N_ft):
			y_imed[k,:] = xbar_dft @ X_image[:,k,ch]
		for k in range(N_ft):
			Y[k,:,ch] = xbar_dft @ y_imed[:,k]
			
	# Inverse 2D DFT
	for ch in range(N_ch):
		x_imed = xp.zeros((Y.shape[1],Y.shape[0]),dtype=np.complex128)
		for k in range(N_ft):
			x_imed[k,:] = xbar_idft @ Y[:,k,ch]
		for k in range(N_ft):
			X_recon[k,:,ch] = xbar_idft @ x_imed[:,k]

###### MatMul based implementation
elif implementation == 1:
	
	for ch in range(N_ch):
		y_imed = xbar_dft @ X_image[:,:,ch]
		Y_ch = y_imed @ xbar_dft
		x_imed = xbar_idft @ Y_ch
		X_recon[:,:,ch] = x_imed @ xbar_idft

###### DFT block based implementation
elif implementation == 2:
	for ch in range(N_ch):
		X_recon[:,:,ch] = dftCore_inv.dft_2d(dftCore_fwd.dft_2d(X_image[:,:,ch]))

# Normalize
X_recon = xp.real(X_recon / (N_ft*N_ft)).clip(0,1)

if useGPU:
	X_image = cp.asnumpy(X_image)
	X_recon = cp.asnumpy(X_recon)

##########################################
## Visualize results

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(9,5))
ax1.imshow(X_image)
ax2.imshow(X_recon)
ax1.set_title("Original image",fontname="Arial",fontsize=16)
ax2.set_title("Reconstructed image",fontname="Arial",fontsize=16)
plt.savefig("./outputs/"+image_id+"_2d_reconstruction.png",bbox_inches="tight",dpi=300)