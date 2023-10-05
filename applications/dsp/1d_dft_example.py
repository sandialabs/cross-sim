#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys
import numpy as np
sys.path.append("../..")
from simulator.algorithms.dsp.dft import DFT
from applications.mvm_params import set_params
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='Arial',style='normal')

"""
This script runs a simple 1D Discrete Fourier Transform demo
This shows an alternative way to use CrossSim to implement DFTs than in
the tutorial, by creating a DFT object that wraps around a CrossSim
AnalogCore. This object is in /simulator/algorithms/dsp/dft.py
"""

np.random.seed(95)

# DFT length
N_ft1 = 512

##########################################
## Set up input images

# Input 1: Generate a narrow rectangular pulse input
rect_real = np.zeros(N_ft1)
i1 = int(np.round(0.48*N_ft1))
i2 = int(np.round(0.52*N_ft1))
rect_real[i1:i2] = 1
rect_imag = np.zeros(N_ft1)
x1 = rect_real + 1j*rect_imag

# Input 2: Generate a sinc pulse input
inds = np.arange(N_ft1)
x = 48*np.pi*(inds-N_ft1/2)/N_ft1 + 1e-16
sinc_real = np.sin(x)/(x)
sinc_real[np.isnan(sinc_real)] = 1
sinc_imag = np.zeros(N_ft1)
x2 = sinc_real + 1j*sinc_imag

###########################################
## Parameters
params = set_params(
	complex_input = True,
	complex_matrix = True,
	wtmodel = "BALANCED",
	error_model = "generic",
	alpha_error = 0.03,
	proportional_error = False,
	noise_model = "none",
	alpha_noise = 0.00,
	proportional_noise = True,
	t_drift = 0,
	drift_model = None,
	Rmin = 1e3,
	Rmax = 1e6,
	infinite_on_off_ratio = True,
	Rp_row = 1e-4,
	Rp_col = 1e-4,
	NrowsMax = 64,
	NcolsMax = 64,
	weight_bits = 8,
	adc_bits = 0,
	adc_range = (0,1),
	dac_bits = 8,
	dac_range = (-1,1),
	interleaved_posneg = False,
	gate_input = False,
	Nslices = 1,	
	digital_offset = False,
	adc_range_option = "CALIBRATED",
	Icol_max = 1e6,
	useGPU = False,
	gpu_id = 0,
	balanced_style = "ONE_SIDED",
	input_bitslicing = True,
	ADC_per_ibit = False)

##########################################
## Create core
dft_core = DFT(N_ft1, params=params)

##########################################
## Process DFTs

y_xbar_1 = dft_core.dft_1d(x1)
y_ideal_1 = np.fft.fftn(x1)

y_xbar_2 = dft_core.dft_1d(x2)
y_ideal_2 = np.fft.fftn(x2)

##########################################
## Graph real and ideal Fourier transform results

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(7,6))

C = np.max(np.abs(np.fft.fftshift(y_ideal_1)))
ax1.plot(C*np.real(x1),'--',color='gray',label='Input')
ax1.plot(np.abs(np.fft.fftshift(y_ideal_1)),label='Ideal')
ax1.plot(np.abs(np.fft.fftshift(y_xbar_1)),label='Crossbar')
leg=ax1.legend(prop=font,loc="upper right",labelspacing=0.2,columnspacing=0.5,handlelength=1.55,handletextpad=0.4)
leg.get_frame().set_edgecolor('k')
ax1.set_ylabel("DFT magnitude",fontname="Arial")

C = np.max(np.abs(np.fft.fftshift(y_ideal_2)))
ax2.plot(C*np.real(x2),'--',color='gray',label='Input')
ax2.plot(np.abs(np.fft.fftshift(y_ideal_2)),label='Ideal')
ax2.plot(np.abs(np.fft.fftshift(y_xbar_2)),label='Crossbar')
leg=ax2.legend(prop=font,loc="upper right",labelspacing=0.2,columnspacing=0.5,handlelength=1.55,handletextpad=0.4)
leg.get_frame().set_edgecolor('k')
ax2.set_xlabel("Element index",fontname="Arial")
ax2.set_ylabel("DFT magnitude",fontname="Arial")

plt.savefig("./outputs/result_1d_dft.png",bbox_inches="tight",dpi=300)