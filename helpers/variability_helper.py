#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from warnings import warn
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from matplotlib import pyplot as plt
import pkg_resources, os
import random


#getting data
def variability_helper(i, inc_file, dec_file, Gmax, CDF_vec_inc,G_vec_inc,dG_matrix_inc,dG_matrix_decr):
	#print("procesing lookup up table=" + str(i +1))
	min_g = G_vec_inc[0]
	max_g = G_vec_inc[-1]
	#print("The lookup table has a physical range of "+str(min_g*1e6)+" uS to "+str(max_g*1e6)+" uS")
	w_vec = G_vec_inc/Gmax
	w_vec_min = w_vec[0]
	w_vec_max = w_vec[-1]
	#print("The lookup table range clipped is "+str(w_vec_min)+" uS to "+str(w_vec_max)+" uS. \nThe xbar weight limits will be rescaled accordingly")
	dw_matrix_inc = dG_matrix_inc/Gmax
	dw_matrix_dec = dG_matrix_decr/Gmax
	#doing some basic transformations to get probability vec
	idx_min = (np.abs(w_vec-w_vec[0])).argmin()
	idx_max = (np.abs(w_vec-w_vec[-1])).argmin()
	if idx_max<(len(w_vec)-1):
		idx_max+=1
	PDF_vec = np.diff(CDF_vec_inc)
	PDF_matrix = np.tile(PDF_vec,[len(w_vec),1])
	PDF_matrix = np.transpose(PDF_matrix)  # 1st index = CDF value, 2nd index = G values (after transpose)
	ave_dw_inc =np.sum(PDF_matrix[:,idx_min:idx_max]*dw_matrix_inc[0:-1,idx_min:idx_max])/(idx_max-idx_min)
	ave_dw_dec =np.sum(PDF_matrix[:,idx_min:idx_max]*dw_matrix_dec[0:-1,idx_min:idx_max])/(idx_max-idx_min)
	if ave_dw_inc<0:
		if ave_dw_dec>0:
			warn("The set and reset files are reversed, flipping order & swapping lookup tables")
			#temp_file = inc_file
			#inc_file = file_dec
			#file_dec = temp_file
			#return manual_post_set()
		else:
			warn("The increasing / SET lookup table has an average update that is negative")
	else:
		if ave_dw_dec>0:
				warn("The decreasing / RESET lookup table has an average update that is positive")
		
	# create average update for each weight value
	ave_dw_inc_vec =np.sum(PDF_matrix*dw_matrix_inc[0:-1,:],axis=0)
	ave_dw_dec_vec =np.sum(PDF_matrix*dw_matrix_dec[0:-1,:],axis=0)
	# create an error matrix by subtracting the mean value for each G from dw_matrix
	dw_error_matrix_dec = dw_matrix_dec-ave_dw_dec_vec
	dw_error_matrix_inc = dw_matrix_inc-ave_dw_inc_vec

	return  ave_dw_inc, ave_dw_dec, ave_dw_inc_vec, ave_dw_dec_vec,dw_error_matrix_dec,dw_error_matrix_inc




