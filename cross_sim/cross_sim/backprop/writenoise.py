#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Write noise models, implemented numerically

from math import sqrt
import numpy as np

STYLES = ("GAUSS","GAUSSCLIP")
GAUSS,GAUSSCLIP = (0,1)
WTMODELS = ("POSNEG","POS")
POSNEG,POS = (0,1)
SCALES = ("FIXED","LOHI","MIDPT","SQRTFIXED","SQRTLOHI","SQRTHILO")
FIXED,LOHI,MIDPT,SQRTFIXED,SQRTLOHI,SQRTHILO = (0,1,2,3,4,5)

BIGVALUE = 1.0e20

# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# numeric WriteNoise models

class WriteNoise:
  """
  write noise models
  required args:
    style = GAUSS,GAUSSCLIP
      GAUSS = Guassian spread of weight change
      GAUSSCLIP = clipped Gaussian spread of weight change
    sigma = size of spread
    clip = +/- clip window on weights for POSNEG wtmodel
    clipmin = min weight value for POS wtmodel
    clipmax = max weight value for POS wtmodel
  optional args:
    wtmodel = POSNEG,POS (def = POSNEG)
      POSNEG = weights are postive and negative
      POS = weights are only positive
    scale = FIXED,LOHI,MIDPT,SQRTFIXED,SQRTLOHI,SQRTHILO (def = FIXED)
      FIXED = noise is independent of weight or delta
      LOHI = noise scales with weight+delta from lo to hi across range
      MIDPT = noise scales with weight+delta from midpt of range
      SQRTFIXED = noise scales with sqrt(delta), independent of weight
      SQRTLOHI = noise scales with sqrt(delta), prop to weight from lo to hi
      SQRTHILO = noise scales with sqrt(delta), prop to weight from hi to lo
      add FUNC = user function for delta scaling ??
    clipgauss = clip bounds on Gaussian RN (def = 1.0)
      only used for style = GAUSSCLIP
    allowflip = 1/0 for whether delta can cross range midpt (def = 1)
      this is effectively flipping numeric sign of weight
    truncate = 1/0 for whether to clip updated wt at range boundaries (def = 1)

  Wnoise = W + delta
  delta = RN * sigmascaled * f(W)
  
  RN = random number
    for GAUSS model: RN = Gaussian with zero mean and unit sigma
    for GAUSSCLIP model: RN = Gaussian, apply clipgauss to avoid sampling tails
    
  sigmascaled = sigma * (hi-lo)
    for POSNEG: lo = -clip, hi = clip
    for POS: lo = clipmin, hi = clipmax

  f(W) = function of weight W
    LOHI and MIDPT options use f(W) with W = matrix + delta
      means the noise depends only on the final updated value
      means the noise will be non-zero even if delta = 0.0
    3 SQRT options use f(W) with W = abs(delta)
      means the noise depends only on the delta change
      means the noise will be 0.0 if delta = 0.0
      some options also depend on M = current matrix wt w/out update
    FIXED: f(W) = 1
    LOHI = scaled from lo to hi
      for POS: f(W) = W from clipmin to clipmax
      for POSNEG: f(W) = alpha W + beta = conversion of POSNEG wt to POS wt
        alpha = (clipmax-clipmin) / (2*clip) = ratio of two ranges
        beta = 0.5 * (clipmin+clipmax) = midpt of POS weight range
    MIDPT = scaled in both directions from midpt of lo to hi
      for POS: f(W) = 2 * (W-clipmid) / (clipmax-clipmin)
        prefactor 2 makes f(W) = 1 at endpts of range, similar to LOHI
      for POSEG: f(W) = W/clip
    SQRTFIXED: f(W) = sqrt(W) / sqrt(cliprange)
    SQRTLOHI = scaled from lo to hi
      for POS: f(W) = sqrt(W)/sqrt(cliprange) * (M/clipmid)
      for POSNEG: f(W) = sqrt(W)/sqrt(cliprange) * ((alpha M + beta) / clipmid)
        2nd terms is conversion of POSNEG wt to POS wt
        alpha = (clipmax-clipmin) / (2*clip) = ratio of two ranges
        beta = 0.5 * (clipmin+clipmax) = midpt of POS weight range
    SQRTHILO = scaled from hi to lo
      for POS: f(W) = sqrt(W)/sqrt(cliprange) * (clipmid/M)
      for POSNEG: f(W) = sqrt(W)/sqrt(cliprange) * (clipmid / (alpha M + beta))
        same details on alpha,beta as for SQRTLOHI
  """

  def __init__(self,**kwargs):
    if "style" not in kwargs: error("Style setting required")
    if "sigma" not in kwargs: error("Sigma setting required")
    if "clip" not in kwargs: error("Clip setting required")
    if "clipmin" not in kwargs: error("Clipmin setting required")
    if "clipmax" not in kwargs: error("Clipmax setting required")
    self.style = kwargs["style"]
    self.sigma = kwargs["sigma"]
    self.clip = kwargs["clip"]
    self.clipmin = kwargs["clipmin"]
    self.clipmax = kwargs["clipmax"]

    # optional args

    self.wtmodel = "POSNEG"
    self.scale = "FIXED"
    self.clipgauss = 1.0
    self.allowflip = 1
    self.truncate = 1
    if "wtmodel" in kwargs: self.wtmodel = kwargs["wtmodel"]
    if "scale" in kwargs: self.scale = kwargs["scale"]
    if "clipgauss" in kwargs: self.clipgauss = kwargs["clipgauss"]
    if "allowflip" in kwargs: self.allowflip = kwargs["allowflip"]
    if "truncate" in kwargs: self.truncate = kwargs["truncate"]

    self.inactive = 0
    if self.style == "NONE" or self.sigma == 0.0:
      self.inactive = 1
      return
    
    # error checks

    if self.style not in STYLES: error("Unknown style")
    self.style = STYLES.index(self.style)
    if self.wtmodel not in WTMODELS: error("Unknown wtmodel")
    self.wtmodel = WTMODELS.index(self.wtmodel)
    if self.scale not in SCALES: error("Unknown scale")
    self.scale = SCALES.index(self.scale)

    if self.sigma < 0.0: raise ValueError("Sigma cannot be < 0.0")
    if self.clip <= 0.0: error("Clip cannot be <= 0.0")
    if self.clipmin <= 0.0: error("Clipmin cannot be <= 0.0")
    if self.clipmax > 1.0: error("Clipmax cannot be > 1.0")
    if self.clipmin >= self.clipmax: error("Clipmin must be < clipmax")
    if self.style == GAUSSCLIP and self.clipgauss <= 0.0:
      error("Clipgauss cannot be <= 0.0")
    if self.allowflip not in (0,1): error("Allowflip must be 0/1")
    if self.truncate not in (0,1): error("Truncate must be 0/1")

    # setup

    if self.wtmodel == POSNEG:
      self.sigmascaled = self.sigma * (2.0*self.clip)
      self.sqrtinvcliprange = sqrt(1.0 / (2.0*self.clip))
    elif self.wtmodel == POS:
      self.sigmascaled = self.sigma * (self.clipmax-self.clipmin)
      self.sqrtinvcliprange = sqrt(1.0 / (self.clipmax - self.clipmin))

    self.alpha = (self.clipmax-self.clipmin) / (2.0*self.clip)
    self.beta = 0.5 * (self.clipmin + self.clipmax)
    self.invclip = 1.0 / self.clip
    self.lohipre = 2.0 / (self.clipmax - self.clipmin)

  # apply noise to delta, which caller will add to matrix of weights
  # does NOT modify matrix, does MODIFY delta

  def apply(self,matrix,delta,debug=0):
    nrow,ncol = matrix.shape
    style = self.style
    sigmascaled = self.sigmascaled
    wtmodel = self.wtmodel
    scale = self.scale

    # rmatrix = matrix of random numbers, based on style
      
    if style == GAUSS:
      RNmatrix = np.random.randn(nrow,ncol)
    elif style == GAUSSCLIP:
      RNmatrix = np.random.randn(nrow,ncol)
      np.clip(RNmatrix,-self.clipgauss,self.clipgauss,out=RNmatrix)

    # noise = perturbation to apply to each weight
    # noise = sigmascaled * RNmatrix * f(W)
    # update delta by noise
      
    if scale == FIXED:
      fW = 1
    elif scale == LOHI:
      matrix_updated = matrix + delta
      if wtmodel == POS: fW = matrix_updated
      else: fW = self.alpha * matrix_updated + self.beta
    elif scale == MIDPT:
      matrix_updated = matrix + delta
      if wtmodel == POS: fW = self.lohipre * (matrix_updated - self.beta)
      else: fW = self.invclip * matrix_updated
    elif scale == SQRTFIXED:
      fW = np.sqrt(np.abs(delta))
      fW *= self.sqrtinvcliprange
    elif scale == SQRTLOHI:
      fW = np.sqrt(np.abs(delta))
      fW *= self.sqrtinvcliprange / self.beta
      if wtmodel == POS: fW *= matrix
      else:
        matrix_converted = self.alpha * matrix + self.beta
        fW *= matrix_converted
    elif scale == SQRTHILO:
      fW = np.sqrt(np.abs(delta))
      fW *= self.sqrtinvcliprange * self.beta
      if wtmodel == POS: fW /= matrix
      else:
        matrix_posneg2pos = self.alpha * matrix + self.beta
        fW /= matrix_posneg2pos
        
    noise = sigmascaled * RNmatrix * fW
    delta += noise

    #ii = 17
    #jj = 43
    #if ncol == 64:
    #  print("MATRIX",matrix[ii][jj])
    #  print("MAT NOISE",noise[ii][jj],delta[ii][jj])

    # if allowflip = 0, do not allow delta to change sign of weight
    # for POSNEG sign-change is at 0.0, for POS sign-change is at beta
    # matrix_update = updated matrix that meets no-flip constraint
    # constrained delta = matrix_update - original matrix
      
    if not self.allowflip:
      if wtmodel == POS: matrix -= self.beta
      posmatrix = np.clip(matrix,0.0,BIGVALUE)
      posmatrix += delta * np.sign(posmatrix)
      np.clip(posmatrix,0.0,BIGVALUE,out=posmatrix)
      negmatrix = np.clip(matrix,-BIGVALUE,0.0)
      negmatrix -= delta * np.sign(negmatrix)
      np.clip(negmatrix,-BIGVALUE,0.0,out=negmatrix)
      matrix_update = np.add(negmatrix,posmatrix)
      np.subtract(matrix_update,matrix,out=delta)
      if wtmodel == POS: matrix += self.beta

    #if ncol == 64:
    #  print("MAT DONE",matrix[ii][jj],delta[ii][jj],
    #        matrix[ii][jj]+delta[ii][jj])

    # NOTE: now do this in caller
    # if truncate = 1, do not allow delta to move weight beyond clip range
    # matrix_update = updated matrix that can be clipped 
    # constrained delta = matrix_update - original matrix
    
    #if self.truncate:
    #  matrix_update = matrix + delta
    #  if wtmodel == POSNEG:
    #    np.clip(matrix_update,-self.clip,self.clip,out=matrix_update)
    #  else:
    #    np.clip(matrix_update,self.clipmin,self.clipmax,out=matrix_update)
    #  np.subtract(matrix_update,matrix,out=delta)

        
