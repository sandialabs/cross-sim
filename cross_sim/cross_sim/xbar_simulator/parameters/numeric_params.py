#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__authors__ = 'sagarwa', 'cbennet', 'txiao'

import pkg_resources, os, random, time
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d, RectBivariateSpline
from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .valueconstraints import NormalError
from .parameter_defaults import WriteNoiseModelEnum, UpdateModelEnum, CrossbarTypeEnum


class WriteNoiseParams(ParametersBase):
	"""
	parameters for write noise
	"""
	if False:
		sigma = float

	write_noise_model = Parameter(name="write_noise_model",
								  post_set=ParametersBase.generate_enum_post_set("write_noise_model",
																				 WriteNoiseModelEnum))

	def __init__(self, param_root,**kwargs):
		#all parameters must be passed in via NumericParams
		ParametersBase.__init__(self,param_root, **kwargs)


class NonlinearityParams(ParametersBase):
	"""
	parameters for write noise
	"""
	if False:
		alpha = float
		symmetric = bool

	def manual_post_set(self): # must call this function manually after setting all parameters to set readonly parameters
		# (this allows values to be computed only once and used multiple times)

		self.override_readonly = True

		params = self.param_root
		alpha = params.numeric_params.nonlinearity.alpha
		wmax = params.xbar_params.weights.maximum
		wmin = params.xbar_params.weights.minimum

		if alpha !=0:
			self.asym_w0 = (wmax-wmin) / (1.0 - np.exp(-alpha))
			self.sym_A = (wmax-wmin) *(np.exp(alpha)+1)/(np.exp(alpha)-1)
			self.sym_B = -(wmax-wmin)/(np.exp(alpha)-1)+wmin

		self.override_readonly = False

	# set readonly vars
	# asymmetric nonlinear prefactor
	asym_w0 = Parameter(name="asym_w0", readonly=True)

	# symmetric nonlinearity model constants
	sym_A = Parameter(name="sym_A", readonly=True)
	sym_B = Parameter(name="sym_B", readonly=True)

	def __init__(self, param_root,**kwargs):
		#all parameters must be passed in via NumericParams
		ParametersBase.__init__(self,param_root, **kwargs)


class NumericParams(ParametersBase):

	if False: # define parameters for easy code completion
		read_noise = NormalError
		write_noise = WriteNoiseParams
		nonlinearity = NonlinearityParams
		dG_lookup = dGLookupParams
		decay = WeightDecayParams
		circuit = CircuitParams

	# use parameter for enums to allow setting with strings
	update_model = Parameter(name="update_model",
							 post_set=ParametersBase.generate_enum_post_set("update_model", UpdateModelEnum))

	def __init__(self, param_root):
		#load the defaults
		attributes = params.NumericParamDefaults['attributes'].copy()
		attributes['read_noise']=NormalError(param_root,**params.NumericParamDefaults['read_noise'])
		attributes['write_noise'] = WriteNoiseParams(param_root,**params.NumericParamDefaults['write_noise'])
		attributes['nonlinearity'] = NonlinearityParams(param_root,**params.NumericParamDefaults['nonlinearity'])
		attributes['dG_lookup'] = dGLookupParams(param_root,**params.NumericParamDefaults['dG_lookup'])
		attributes['circuit'] = CircuitParams(param_root,**params.NumericParamDefaults['circuit'])

		ParametersBase.__init__(self,param_root, **attributes)


class CircuitParams(ParametersBase):
	"""
	parameters for read and write circuits
	"""
	if False:
		VrowS = float
		VrowUS = float
		VcolUS = float
		Vselect = float
		Vread = float
		mvm_encoding = str

	def __init__(self, param_root,**kwargs):
		#all parameters must be passed in via NumericParams
		ParametersBase.__init__(self,param_root, **kwargs)


class dGLookupParams(ParametersBase):

	"""
	parameters for dG look up table
	"""
	
	# For uniform LUT
	Gmax_clip = Parameter(name="Gmax_clip", readonly=True)  # store the absolute max and min conductance for reference
	Gmin_clip = Parameter(name="Gmin_clip", readonly=True)  # store the absolute max and min conductance for reference
	w_vec = Parameter(name="w_vec", readonly=True)  #vector of weights for the columns of the lookup table
	CDF_vec = Parameter(name="CDF_vec", readonly=True)  # CDF vector for the rows of the lookup table
	ave_dw_inc = Parameter(name="ave_dw_inc", readonly=True)  # average update in the lookup table for increasing pulses
	ave_dw_dec = Parameter(name="ave_dw_dec", readonly=True)  # average update in the lookup table for decreasing pulses
	dw_matrix_inc = Parameter(name="dG_matrix_inc", readonly=True)  # lookup table of delta weight for positive pulses
	dw_matrix_dec = Parameter(name="dG_matrix_dec", readonly=True)  # lookup table of delta weight for negative pulses
	ave_dw_inc_vec = Parameter(name="ave_dw_inc_vec", readonly=True)  # average update for each weight value, pos pulses
	ave_dw_dec_vec = Parameter(name="ave_dw_dec_vec", readonly=True)  # average update for each weight value, neg pulses
	dw_error_matrix_dec = Parameter(name="dw_error_matrix_dec",readonly=True)  # lookup table of noise added to each pulse, neg pulses, assuming mean update above
	dw_error_matrix_inc = Parameter(name="dw_error_matrix_inc",readonly=True)  # lookup table of noise added to each pulse, pos pulses, assuming mean update above
	error_pos_interpolator = Parameter(name="error_pos_interpolator",readonly=True)  # interpolation functions defined once at the beginning
	error_neg_interpolator = Parameter(name="error_neg_interpolator",readonly=True)  # interpolation functions defined once at the beginning
	ave_dw_pos_interpolator = Parameter(name="ave_dw_pos_interpolator",readonly=True)  # interpolation functions defined once at the beginning
	ave_dw_neg_interpolator = Parameter(name="ave_dw_neg_interpolator",readonly=True)  # interpolation functions defined once at the beginning
	slow_interpolator = Parameter(name="slow_interpolator",readonly=True)  # whether a slow SciPy interpolator is used (called differently)

	# For multi-LUT
	matsize = Parameter(name="matsize", readonly=False)  # holds dim for am1
	gbinsmat = Parameter(name="gbinsmat", readonly=False)  # device config
	CDF_vec_lutmat = Parameter(name="CDF_vec_lutmat", readonly=False)
	CDF_step = Parameter(name="CDF_step", readonly=False)
	ave_dw_dec_vec_lutmat = Parameter(name="ave_dw_dec_vec_lutmat", readonly=False)  # device config
	ave_dw_inc_vec_lutmat = Parameter(name="ave_dw_inc_vec_lutmat", readonly=False)  # device config
	ave_dw_dec_lutmat = Parameter(name="ave_dw_dec_lutmat", readonly=False)  # device config
	ave_dw_inc_lutmat = Parameter(name="ave_dw_inc_lutmat", readonly=False)  # device config
	dw_error_dec_lutmat = Parameter(name="dw_error_dec_lutmat", readonly=False)  # device config
	dw_error_inc_lutmat = Parameter(name="dw_error_inc_lutmat", readonly=False)  # device config
	w_steps_lut_vec = Parameter(name="w_steps_lut_vec", readonly=False)  # device config
	dw_bools = Parameter(name="dw_bools", readonly=False)  

	w_mins_LUT = Parameter(name="w_mins_LUT", readonly=False)  
	T_vec = Parameter(name="T_vec", readonly=False)  

	w_mins = Parameter(name="w_mins", readonly=False)  
	w_ranges = Parameter(name="w_ranges", readonly=False)  
	N_updates = Parameter(name="N_updates", readonly=False)  
	
	if False:
		file_increasing = str
		file_decreasing = str
		Gmin_relative = float
		Gmax_relative = float
		disable_nonlinearity =bool # can be changed dynamically for calibrated updates
		disable_writenoise=bool # can be changed dynamically


	def __init__(self, param_root,**kwargs):
		#all parameters must be passed in via NumericParams
		ParametersBase.__init__(self,param_root, **kwargs)


	def manual_post_set(self):

		# If Single-LUT, LUT files need to be read and processed; read-only parameters will be set
		# If Multi-LUT, LUT files have already been read and processed; here, assign device LUTs to array elements
		params = self.param_root

		# exit post_set unless using dG lookup model
		if params.numeric_params.update_model != UpdateModelEnum.DG_LOOKUP:
			return

		global ncp
		if self.gpu_engaged:
			global cp
			import cupy as cp
			cp.cuda.Device(0).use()
			ncp = cp
		else:
			ncp = np

		self.N_updates = 0

		if self.multi_LUT:
			# Generate dw_bools: set of boolean matrices of array elements that use the LUT
			self.Nluts = self.file_decreasing.shape[2]
			self.table_dim = self.gbinsmat.shape[0]
			self.dw_bools = np.full((self.matsize[0],self.matsize[1],self.Nluts),False,dtype=bool)
			assignment_mat = np.random.choice([x for x in range(self.Nluts)],self.matsize)
			for k in range(self.Nluts):
				dw_bool_k = np.full(self.matsize,False,dtype=bool)
				dw_bool_k[tuple(np.array(np.where(assignment_mat==k)))] = True
				self.dw_bools[:,:,k] = dw_bool_k.copy()

			# Pre-compute CDF stepsize: the rows of this matrix are actually all the same
			self.CDF_step = np.mean(np.diff(self.CDF_vec_lutmat[:,1]))

			if self.Nlevels > 0:
				w_mins = np.zeros((self.matsize[0],self.matsize[1]))
				w_ranges = np.zeros((self.matsize[0],self.matsize[1]))
				for k in range(self.Nluts):
					w_mins += self.dw_bools[:,:,k] * self.w_limits[k,0]
					w_ranges += self.dw_bools[:,:,k] * (self.w_limits[k,1] - self.w_limits[k,0])
				self.w_mins = w_mins
				self.w_ranges = w_ranges

			if self.gpu_engaged:
				self.ave_dw_inc_vec_lutmat = cp.array(self.ave_dw_inc_vec_lutmat)
				self.ave_dw_dec_vec_lutmat = cp.array(self.ave_dw_dec_vec_lutmat)
				self.dw_error_dec_lutmat = cp.array(self.dw_error_dec_lutmat)
				self.dw_error_inc_lutmat = cp.array(self.dw_error_inc_lutmat)
				self.ave_dw_inc_lutmat = cp.array(self.ave_dw_inc_lutmat)
				self.ave_dw_dec_lutmat = cp.array(self.ave_dw_dec_lutmat)
				self.dw_bools = cp.array(self.dw_bools)
				self.w_steps_lut_vec = cp.array(self.w_steps_lut_vec)
				self.gbinsmat = cp.array(self.gbinsmat)
				if self.Nlevels > 0:
					self.w_mins = cp.array(self.w_mins)
					self.w_ranges = cp.array(self.w_ranges)

			# If using vectorized code, flatten all the LUTs to allow parallel indexing
			if self.vectorize:
				self.ave_dw_inc_vec_lutmat = self.ave_dw_inc_vec_lutmat.flatten()
				self.ave_dw_dec_vec_lutmat = self.ave_dw_dec_vec_lutmat.flatten()
				self.dw_error_dec_lutmat = self.dw_error_dec_lutmat.flatten()
				self.dw_error_inc_lutmat = self.dw_error_inc_lutmat.flatten()				
				self.w_mins_LUT = self.gbinsmat[0,:] - self.w_steps_lut_vec/2

			return

		# Process lookup table files: this is done at this step for single LUT only
		self.override_readonly = True

		# load lookup tables, try filename and then search stored data
		if os.path.isfile(self.file_increasing):
			filename_increasing = self.file_increasing
		elif os.path.isfile(pkg_resources.resource_filename("cross_sim","data/lookup_tables/" +self.file_increasing+"/dG_increasing.txt")):
			filename_increasing = pkg_resources.resource_filename("cross_sim","data/lookup_tables/" +self.file_increasing+"/dG_increasing.txt")
		else:
			raise FileNotFoundError("The dG lookup table file is not valid: "+str(self.file_increasing))
		if os.path.isfile(self.file_decreasing):
			filename_decreasing = self.file_decreasing
		elif os.path.isfile(pkg_resources.resource_filename("cross_sim","data/lookup_tables/" +self.file_decreasing+"/dG_decreasing.txt")):
			filename_decreasing = pkg_resources.resource_filename("cross_sim","data/lookup_tables/" +self.file_decreasing+"/dG_decreasing.txt")
		else:
			raise FileNotFoundError("The dG lookup table file is not valid: "+str(self.file_decreasing))

		# load data
		G_vec_inc = np.genfromtxt(filename_increasing,delimiter=',',skip_header=1, max_rows=1)  # max_rows option requires numpy v1.10 or higher
		CDF_vec_inc = np.genfromtxt(filename_increasing,delimiter=',',skip_header=2, max_rows=1)  # max_rows option requires numpy v1.10 or higher
		dG_matrix_inc = np.genfromtxt(filename_increasing,delimiter=',',skip_header=3)
		G_vec_dec = np.genfromtxt(filename_decreasing,delimiter=',',skip_header=1, max_rows=1)  # max_rows option requires numpy v1.10 or higher
		CDF_vec_dec = np.genfromtxt(filename_decreasing,delimiter=',',skip_header=2, max_rows=1)  # max_rows option requires numpy v1.10 or higher
		dG_matrix_dec = np.genfromtxt(filename_decreasing,delimiter=',',skip_header=3)

		if (G_vec_inc != G_vec_dec).all(): raise ValueError("The increasing and decreasing dG lookup files must have the same G values")
		if (CDF_vec_inc != CDF_vec_dec).all(): raise ValueError("The increasing and decreasing dG lookup files must have the same CDF values")

		# Store the absolute Gmax and Gmin for reference
		# Then rescale conductances and store the LUT conductance vector
		self.Gmax_clip = G_vec_inc[-1]
		self.Gmin_clip = G_vec_inc[0]
		Gmax = self.Gmin_clip + (self.Gmax_clip-self.Gmin_clip)*self.Gmax_relative
		Gmin = self.Gmin_clip + (self.Gmax_clip-self.Gmin_clip)*self.Gmin_relative
		self.w_vec = G_vec_inc/Gmax

		params.xbar_params.weight_clipping.minimum = self.w_vec[0]
		params.xbar_params.weight_clipping.maximum = self.w_vec[-1]
		params.xbar_params.weights.minimum = Gmin/Gmax
		params.xbar_params.weights.maximum = 1

		# store the CDF and normalized dG arrays
		self.CDF_vec = CDF_vec_inc
		self.dw_matrix_inc = dG_matrix_inc/Gmax
		self.dw_matrix_dec = dG_matrix_dec/Gmax

		## Find the average update, weighted by the probability

		# find indices corresponding to the range used so that average is only over used range
		idx_min = np.abs(self.w_vec-params.xbar_params.weights.minimum).argmin()
		idx_max = np.abs(self.w_vec-params.xbar_params.weights.maximum).argmin()
		if idx_max < len(self.w_vec)-1:
			idx_max+=1 # add one to include endpoint: x[a:b] does not include b

		# create vector of probabilities
		PDF_vec = np.diff(self.CDF_vec)
		PDF_matrix = np.tile(PDF_vec,[len(self.w_vec),1])
		PDF_matrix = np.transpose(PDF_matrix)  # 1st index = CDF value, 2nd index = G values (after transpose)
		self.ave_dw_inc =np.sum(PDF_matrix[:,idx_min:idx_max]*self.dw_matrix_inc[0:-1,idx_min:idx_max])/(idx_max-idx_min)
		self.ave_dw_dec =np.sum(PDF_matrix[:,idx_min:idx_max]*self.dw_matrix_dec[0:-1,idx_min:idx_max])/(idx_max-idx_min)

		#check that average updates have the correct sign, flip set and reset if needed.
		if self.ave_dw_inc < 0:
			if self.ave_dw_dec > 0:
				warn("The set and reset files are reversed, flipping order & swapping lookup tables")
				temp_file = self.file_increasing
				self.file_increasing = self.file_decreasing
				self.file_decreasing = temp_file
				return self.manual_post_set()
			else:
				raise ValueError("The increasing / SET lookup table has an average update that is negative")
		else:
			if self.ave_dw_dec > 0:
				raise ValueError("The decreasing / RESET lookup table has an average update that is positive")

		print("The lookup table has a range of {:.4f}".format(self.Gmin_clip*1e6)+" uS to {:.4f}".format(self.Gmax_clip*1e6)+" uS")
		print("The lookup table range used is {:.4f}".format(Gmin*1e6)+" uS to {:.4f}".format(Gmax*1e6)+" uS. \nThe xbar weight limits will be rescaled accordingly")

		# create average update for each weight value
		self.ave_dw_inc_vec = np.sum(PDF_matrix*self.dw_matrix_inc[0:-1,:],axis=0)
		self.ave_dw_dec_vec = np.sum(PDF_matrix*self.dw_matrix_dec[0:-1,:],axis=0)

		# create an error matrix by subtracting the mean value for each G from dw_matrix
		self.dw_error_matrix_dec = self.dw_matrix_dec - self.ave_dw_dec_vec
		self.dw_error_matrix_inc = self.dw_matrix_inc - self.ave_dw_inc_vec

		# create lookup table interpolator if possible, otherwise do linear interpolation
		w_steps = np.diff(self.w_vec)

		# Check to see if LUT has uniform spacing (allow small roundoff errors)
		if (PDF_vec.max()-PDF_vec.min())<PDF_vec.max()/1e4 and (w_steps.max()-w_steps.min() )<w_steps.max()/1e4:
			# find step sizes
			CDF_step = np.mean(PDF_vec)
			w_step = np.mean(w_steps)
			self.slow_interpolator = False

			# define interpolator functions
			def error_pos_interpolator(CDF, w):
				# do interpolation by defining a numpy array indices by rounding and then apply indices to numpy array
				CDF_indices = (CDF+CDF_step/2)/CDF_step
				w_indices = (w -(params.xbar_params.weight_clipping.minimum -w_step/2)  )/w_step
				return self.dw_error_matrix_inc[CDF_indices.astype(int),w_indices.astype(int)]

			def error_neg_interpolator(CDF, w):
				CDF_indices = (CDF+CDF_step/2)/CDF_step
				w_indices = (w -(params.xbar_params.weight_clipping.minimum -w_step/2)  )/w_step
				return self.dw_error_matrix_dec[CDF_indices.astype(int),w_indices.astype(int)]

			# create interpolator for mean dw values
			def ave_dw_pos_interpolator(w):
				w_indices = (w -(params.xbar_params.weight_clipping.minimum - w_step/2)  )/w_step
				return self.ave_dw_inc_vec[w_indices.astype(int)]

			def ave_dw_neg_interpolator(w):
				w_indices = (w -(params.xbar_params.weight_clipping.minimum -w_step/2)  )/w_step
				return self.ave_dw_dec_vec[w_indices.astype(int)]

			self.error_pos_interpolator = error_pos_interpolator
			self.error_neg_interpolator = error_neg_interpolator
			self.ave_dw_pos_interpolator = ave_dw_pos_interpolator
			self.ave_dw_neg_interpolator = ave_dw_neg_interpolator

		else:
			warn("Using slower interpolator, should define dG vs G lookup table on a regular grid")
			if self.gpu_engaged:
				raise ValueError("GPU acceleration not yet supported for non-uniform lookup tables")
			self.slow_interpolator = True
			self.error_pos_interpolator = RectBivariateSpline(self.CDF_vec,self.w_vec, self.dw_error_matrix_inc, kx=1,ky=1)
			self.error_neg_interpolator = RectBivariateSpline(self.CDF_vec,self.w_vec, self.dw_error_matrix_dec, kx=1,ky=1)

			# create interpolator for mean dw values
			self.ave_dw_pos_interpolator = interp1d(self.w_vec, self.ave_dw_inc_vec, fill_value=(self.ave_dw_inc_vec[0],self.ave_dw_inc_vec[-1]), bounds_error=False)
			self.ave_dw_neg_interpolator = interp1d(self.w_vec, self.ave_dw_dec_vec, fill_value=(self.ave_dw_dec_vec[0],self.ave_dw_dec_vec[-1]), bounds_error=False)


		if self.gpu_engaged:
			self.ave_dw_inc_vec = cp.array(self.ave_dw_inc_vec)
			self.ave_dw_dec_vec = cp.array(self.ave_dw_dec_vec)
			self.dw_error_matrix_inc = cp.array(self.dw_error_matrix_inc)
			self.dw_error_matrix_dec = cp.array(self.dw_error_matrix_dec)

		self.override_readonly = False


	def compute_update(self, dw, w, disable_writenoise=None, disable_nonlinearity=None, parallel=False, gpu=False):
		"""

		:param dw: the requested change in weights
		:param w: the current states, should be same size as dw
		:param disable_writenoise:  Use the writenoise in the lookup table.  If True, only apply nonlinearity. If None, use self.disable_writenoise
		:return: update:  the actual update given the nonlinearity and writenoise from the lookup_tablesup table

		"""

		# Intialize writenoise and nonlinearity from params, overwrite if passed in
		if disable_writenoise is None:
			disable_writenoise = self.disable_writenoise
		if disable_nonlinearity is None:
			disable_nonlinearity = self.disable_nonlinearity

		if self.multi_LUT:
			if disable_writenoise and disable_nonlinearity:
				update = dw
			else:
				update = self.lut_variability_update(w, dw, disable_writenoise, disable_nonlinearity)

		else:
			pos_indices  = dw > 0
			neg_indices = ncp.logical_not(pos_indices)

			# interpolate the average update based on weight (compute nonlinearity)
			update = ncp.full(w.shape,ncp.nan)
			update[pos_indices] = self.ave_dw_pos_interpolator(w[pos_indices])
			update[neg_indices] = self.ave_dw_neg_interpolator(w[neg_indices])

			## compute unscaled write noise
			if disable_writenoise:
				dw_error = ncp.zeros(w.shape)
			else:
				dw_error = ncp.full(w.shape,np.nan)

				# Find the error in update (write noise)
				# CDF is a matrix of uniform random CDF values between 0 and 1; interpolator returns the dw error value at each chosen CDF
				CDF = ncp.random.random_sample(w.shape)
				if not self.slow_interpolator:
					dw_error[pos_indices] = self.error_pos_interpolator(CDF[pos_indices], w[pos_indices])
					dw_error[neg_indices] = self.error_neg_interpolator(CDF[neg_indices], w[neg_indices])
				else:
					dw_error[pos_indices] = self.error_pos_interpolator.ev(CDF[pos_indices], w[pos_indices])
					dw_error[neg_indices] = self.error_neg_interpolator.ev(CDF[neg_indices], w[neg_indices])

			scale_mat = ncp.full(w.shape,ncp.nan)

			if disable_nonlinearity:
				# compute update scaled by nonlinearity for noise  (if i.e. adjust update size to cancel out nonlinearity)
				scale_mat = dw/update

				# if the average nonlinearity is the wrong sign at a given w, just use the average update for noise scaling
				error_pos_indices = ncp.logical_and(scale_mat<=0, pos_indices)
				scale_mat[error_pos_indices]= dw[error_pos_indices]/self.ave_dw_inc
				error_neg_indices = ncp.logical_and(scale_mat<=0, neg_indices)
				scale_mat[error_neg_indices]= dw[error_neg_indices]/self.ave_dw_dec

				# Scale the write noise
				dw_error *= ncp.sqrt(scale_mat)
				update = dw + dw_error

			else:
				# find scaling between mean update and target update
				scale_mat[pos_indices] = dw[pos_indices]/self.ave_dw_inc
				scale_mat[neg_indices] = dw[neg_indices]/self.ave_dw_dec

				# scale error, assuming sigma^2 adds
				dw_error *= ncp.sqrt(scale_mat)

				#scale the update and add the error
				update = scale_mat*update + dw_error

		# Quantize weight levels if enabled
		if self.Nlevels > 0:
			w_cont = w + update
			w_quant = w_cont - self.w_mins
			w_quant /= self.w_ranges
			w_quant *= self.Nlevels - 1
			if not self.stochastic_quantization:
				w_quant = ncp.rint(w_quant,out=w_quant)
			else:
				w_quant_floor = ncp.floor(w_quant)

				# rand_mat = ncp.random.sample(w_quant.shape)
				rand_mat = ncp.random.normal(loc=0.5,scale=1.0,size=w_quant.shape)
				w_quant = w_quant_floor + (rand_mat<(w_quant-w_quant_floor))

			w_quant /= self.Nlevels - 1
			w_quant *= self.w_ranges
			w_quant += self.w_mins
			update = w_quant - w

		self.N_updates += 1

		return update

	# Select the appropriate index matrix and apply multi LUT update
	# AT MOST ONE of disable_writenoise and disable_nonlinearity can be true
	def lut_variability_update(self, w, dw, disable_writenoise, disable_nonlinearity):

		if not self.vectorize:
			update, scale_mat, dw_error = self.matrix_update_multiLUT(w,dw,disable_writenoise,disable_nonlinearity)
		else:
			update, scale_mat, dw_error = self.matrix_update_multiLUT_vectorized(w,dw,disable_writenoise,disable_nonlinearity)

		if disable_writenoise:
			update *= scale_mat
		else:
			if disable_nonlinearity:
				update = dw + dw_error*ncp.sqrt(scale_mat)
			else:
				sqrt_scale_mat = ncp.sqrt(scale_mat)

				## To improve performance, assume the following never comes up
				# if ncp.isnan(sqrt_scale_mat).any():
				# 	update = scale_mat*update + dw_error*0.15 # replace if nans
				# else:
				#	update = scale_mat*update + dw_error*sqrt_scale_mat

				update = scale_mat*update + dw_error*sqrt_scale_mat

		return update

	# Compute the multi-LUT matrix update with write noise
	def matrix_update_multiLUT(self, w, dw, disable_writenoise, disable_nonlinearity):

		wmax = self.param_root.xbar_params.weights.maximum
		wmin = self.param_root.xbar_params.weights.minimum
		gbinsmat = self.gbinsmat
		ave_dw_inc = self.ave_dw_inc_lutmat
		ave_dw_dec = self.ave_dw_dec_lutmat
		ave_dw_inc_mat = self.ave_dw_inc_vec_lutmat
		ave_dw_dec_mat = self.ave_dw_dec_vec_lutmat
		w_steps_vec = self.w_steps_lut_vec

		update = ncp.zeros(w.shape)
		scale_mat = ncp.zeros(w.shape)

		if not disable_writenoise:
			dw_error_dec_lutmat = self.dw_error_dec_lutmat
			dw_error_inc_lutmat = self.dw_error_inc_lutmat
			CDF_step = self.CDF_step
			dw_error = ncp.zeros(w.shape)
			CDF = ncp.random.random_sample((w.shape[0],w.shape[1],self.Nluts))
		else:
			dw_error = None

		pos_indices = dw > 0
		neg_indices = ncp.logical_not(pos_indices)
		
		# Don't update weights that are beyond the limits
		w_bool = ncp.full((dw.shape),True,dtype=bool)
		w_bool[w > wmax] = False
		w_bool[w < wmin] = False
		pos_indices = ncp.logical_and(pos_indices,w_bool)
		neg_indices = ncp.logical_and(neg_indices,w_bool)

		for k in range(self.Nluts):
			# If no array element uses the table, skip
			if not self.dw_bools[:,:,k].any(): continue

			w_vec = gbinsmat[:,k]
			w_step = w_steps_vec[k]
			ave_dw_inc_vec = ave_dw_inc_mat[:,k]
			ave_dw_dec_vec = ave_dw_dec_mat[:,k]
			CDF_k = CDF[:,:,k]

			# Select the elements to udpate
			pos_indices_k = ncp.logical_and(self.dw_bools[:,:,k],pos_indices)
			neg_indices_k = ncp.logical_and(self.dw_bools[:,:,k],neg_indices)

			# Get the average updates and scale factors
			update[pos_indices_k] = self.ave_dw_interpolator_var(w[pos_indices_k],w_vec,w_step,ave_dw_inc_vec)
			update[neg_indices_k] = self.ave_dw_interpolator_var(w[neg_indices_k],w_vec,w_step,ave_dw_dec_vec)

			if disable_writenoise:
				scale_mat[pos_indices_k] = dw[pos_indices_k]/ave_dw_inc[0,k]
				scale_mat[neg_indices_k] = dw[neg_indices_k]/ave_dw_dec[1,k]
			else:
				dw_error_matrix_inc = dw_error_inc_lutmat[:,:,k]
				dw_error_matrix_dec = dw_error_dec_lutmat[:,:,k]
				dw_error[pos_indices_k] = self.error_interpolator_var(CDF_k[pos_indices_k],CDF_step,w[pos_indices_k],w_vec,w_step,dw_error_matrix_inc)
				dw_error[neg_indices_k] = self.error_interpolator_var(CDF_k[neg_indices_k],CDF_step,w[neg_indices_k],w_vec,w_step,dw_error_matrix_dec)
				if disable_nonlinearity:
					scale_mat[pos_indices_k] = dw[pos_indices_k]/update[pos_indices_k]
					scale_mat[neg_indices_k] = dw[neg_indices_k]/update[neg_indices_k]
					error_pos_indices = ncp.logical_and(scale_mat<=0, pos_indices_k)
					scale_mat[error_pos_indices] = dw[error_pos_indices]/ave_dw_inc[k]
					error_neg_indices = ncp.logical_and(scale_mat<=0, neg_indices_k)
					scale_mat[error_neg_indices] = dw[error_neg_indices]/ave_dw_dec[k]
				else:
					scale_mat[pos_indices_k] = dw[pos_indices_k]/ave_dw_inc[k]
					scale_mat[neg_indices_k] = dw[neg_indices_k]/ave_dw_dec[k]

		return update, scale_mat, dw_error


	## Interpolator for mean dw values for multi-LUT
	def ave_dw_interpolator_var(self,w,w_vec,w_step,ave_dw_vec):
	    w_indices = (w - (w_vec[0] - w_step/2))/w_step
	    w_indices[w_indices>len(w_vec)-1] = len(w_vec) - 1
	    return ave_dw_vec[w_indices.astype(np.int32)]

	## Interpolator for errors in dw values for multi-LUT
	def error_interpolator_var(self,CDF,CDF_step,w,w_vec,w_step,dw_error_matrix):
	    CDF_indices = (CDF + CDF_step/2)/CDF_step
	    w_indices = (w - (w_vec[0] - w_step/2))/w_step
	    w_indices[w_indices>len(w_vec)-1] = len(w_vec) - 1
	    w_indices[w_indices<0] = 0
	    return dw_error_matrix[CDF_indices.astype(np.int32),w_indices.astype(np.int32)]


	# Compute the multi-LUT matrix update with write noise
	def matrix_update_multiLUT_vectorized(self, w, dw, disable_writenoise, disable_nonlinearity, broadcast=False):
		
		# Interestingly, the 'broadcast' setting does not seem to have much impact on GPU performance

		if disable_writenoise or disable_nonlinearity:
			raise ValueError("Vectorized LUT method currently does not support disabling write noise or nonlinearity.")

		wmax = self.param_root.xbar_params.weights.maximum
		wmin = self.param_root.xbar_params.weights.minimum
		ave_dw_inc = self.ave_dw_inc_lutmat
		ave_dw_dec = self.ave_dw_dec_lutmat
		ave_dw_inc_mat = self.ave_dw_inc_vec_lutmat
		ave_dw_dec_mat = self.ave_dw_dec_vec_lutmat
		w_steps_vec = self.w_steps_lut_vec
		dw_error_dec_lutmat = self.dw_error_dec_lutmat
		dw_error_inc_lutmat = self.dw_error_inc_lutmat
		
		CDF_all = ncp.random.random_sample((w.shape[0],w.shape[1],self.Nluts))

		pos_indices = dw > 0
		neg_indices = ncp.logical_not(pos_indices)
		
		# Don't update weights that are beyond the limits
		w_bool = ncp.logical_and(w <= wmax, w >= wmin)
		pos_indices = ncp.logical_and(pos_indices,w_bool)
		neg_indices = ncp.logical_and(neg_indices,w_bool)
		pos_indices = ncp.logical_and(self.dw_bools,pos_indices[:,:,None])
		neg_indices = ncp.logical_and(self.dw_bools,neg_indices[:,:,None])

		if not broadcast:
			w_all = ncp.repeat(w[:,:,None],self.Nluts,axis=2)
			dw_all = ncp.repeat(dw[:,:,None],self.Nluts,axis=2)
			w_indices = (w_all - self.w_mins_LUT)/w_steps_vec
		else:
			w_indices = (w[:,:,None] - self.w_mins_LUT[None,None,:])/w_steps_vec

		CDF_indices = (CDF_all + self.CDF_step/2)/self.CDF_step
		w_indices = ncp.clip(w_indices,0,self.table_dim-1)
		CDF_indices = ncp.clip(CDF_indices,0,self.table_dim-1)

		# The following code assumes:
		# The un-flattened ave_dw_inc_mat is accessed by the order: (w_indices, lut)
		#	with dimensions (table_dim, Nluts)
		# The un-flatted dw_error_inc_lumat is accessed by the order: (CDF_indices, w_indices, lut)
		#	with dimensions (table_dim, table_dim, Nluts)
		# If any of the above changes, the code needs to be modified

		w_indices *= self.Nluts
		w_indices += ncp.arange(self.Nluts)
		err_indices = CDF_indices*self.table_dim*self.Nluts + w_indices

		w_indices = w_indices.astype(ncp.int32)
		err_indices = err_indices.astype(ncp.int32)

		# Multiplying by pos_indices and neg_indices inside the index reduces the number of unique elements accessed
		# Gives a small net speedup
		update = pos_indices*ave_dw_inc_mat[w_indices*pos_indices] + neg_indices*ave_dw_dec_mat[w_indices*neg_indices]
		update = ncp.sum(update,axis=2)

		dw_error = pos_indices*dw_error_inc_lutmat[err_indices*pos_indices] + neg_indices*dw_error_dec_lutmat[err_indices*neg_indices]
		dw_error = ncp.sum(dw_error,axis=2)

		if not broadcast:
			scale_mat = pos_indices*dw_all/ave_dw_inc + neg_indices*dw_all/ave_dw_dec
		else:
			scale_mat = pos_indices*dw[:,:,None]/ave_dw_inc + neg_indices*dw[:,:,None]/ave_dw_dec
		scale_mat = ncp.sum(scale_mat,axis=2)

		return update, scale_mat, dw_error