#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Read noise models, implemented numerically

import numpy as np

STYLES = ("RTN","UNIFORM","GAUSS","GAUSSCLIP")
RTN,UNIFORM,GAUSS,GAUSSCLIP = (0,1,2,3)
WTMODELS = ("POSNEG","POS")
POSNEG,POS = (0,1)
SCALES = ("FIXED","LOHI","MIDPT")
FIXED,LOHI,MIDPT = (0,1,2)

BIGVALUE = 1.0e20

# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# numeric ReadNoise models

class ReadNoise:
  """
  read noise models
  required args:
    style = RTN,UNIFORM,GAUSS,GAUSSCLIP
      RTN = Random Telegraph Noise two-state model
      UNIFORM = uniform spread of weight change
      GAUSS = Gaussian spread of weight change
      GAUSSCLIP = clipped Gaussian spread of weight change
    sigma  = size of spread
    clip = +/- clip window on weights for POSNEG wtmodel
    clipmin = min weight value for POS wtmodel
    clipmax = max weight value for POS wtmodel
  optional args:
    wtmodel = POSNEG,POS (def = POSNEG)
      POSNEG = weights are positive and negative
      POS = weights are only positive
    scale = FIXED,LOHI,MIDPT (def = FIXED)
      FIXED = delta is independent of weight
      LOHI = delta scales with weight from lo to hi across range
      MIDPT = delta scales with weight from midpt of range
      add FUNC = user function for scaling ??
    clipgauss = clip bounds on Gaussian RN (def = 1.0)
      only used for style = GAUSSCLIP
    allowflip = 1/0 for whether delta can cross range midpt (def = 1)
      this is effectively flipping numeric sign of weight
    truncate = 1/0 for whether to clip updated wt at range boundaries (def = 1)
      
  Wnoise = W + delta
  delta = RN * sigmascaled * f(W)
  
  RN = random number
    for RTN model: RN = -1,1 with 50/50 probability
    for UNIFORM model: RN = uniform from -1.0 to 1.0
    for GAUSS model: RN = Gaussian with zero mean and unit sigma
    for GAUSSCLIP model: RN = Gaussian, apply clipgauss to avoid sampling tails

  sigmascaled = sigma * (hi-lo)
    for POSNEG: lo = -clip, hi = clip
    for POS: lo = clipmin, hi = clipmax

  f(W) = function of weight W
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

    if self.sigma < 0.0: error("Sigma cannot be < 0.0")
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
    elif self.wtmodel == POS:
      self.sigmascaled = self.sigma * (self.clipmax-self.clipmin)

    self.alpha = (self.clipmax-self.clipmin) / (2.0*self.clip)
    self.beta = 0.5 * (self.clipmin + self.clipmax)
    self.invclip = 1.0 / self.clip
    self.lohipre = 2.0 / (self.clipmax - self.clipmin)

  # apply noise to matrix of weights
  # this MODIFIES matrix
  # caller passes copy of permanent matrix, copy is modified

  def apply(self,matrix,debug=0):
    nrow,ncol = matrix.shape
    style = self.style
    sigmascaled = self.sigmascaled
    wtmodel = self.wtmodel
    scale = self.scale
    
    # rmatrix = matrix of random numbers, based on style
      
    if style == RTN:
      RNmatrix = 2.0 * np.random.randint(0,2,(nrow,ncol)) - 1.0
    elif style == UNIFORM:
      RNmatrix = 2.0 * np.random.rand(nrow,ncol) - 1.0
    elif style == GAUSS:
      RNmatrix = np.random.randn(nrow,ncol)
    elif style == GAUSSCLIP:
      RNmatrix = np.random.randn(nrow,ncol)
      np.clip(RNmatrix,-self.clipgauss,self.clipgauss,out=RNmatrix)

    # delta = perturbation to apply to each weight
    # delta = sigmascaled * RNmatrix * f(W)

    if scale == FIXED:
      delta = sigmascaled * RNmatrix
    elif scale == LOHI:
      if wtmodel == POS: fW = matrix
      else: fW = self.alpha * matrix + self.beta
      delta = sigmascaled * RNmatrix * fW
    elif scale == MIDPT:
      if wtmodel == POS: fW = self.lohipre * (matrix - self.beta)
      else: fW = self.invclip * matrix
      delta = sigmascaled * RNmatrix * fW

    # add delta to matrix
    # if allowflip = 0, do not allow delta to change sign of weight
    # for POSNEG sign-change is at 0.0, for POS sign-change is at beta
      
    if self.allowflip: np.add(matrix,delta,out=matrix)
    else:
      if wtmodel == POS: matrix -= self.beta
      posmatrix = np.clip(matrix,0.0,BIGVALUE)
      posmatrix += delta * np.sign(posmatrix)
      np.clip(posmatrix,0.0,BIGVALUE,out=posmatrix)
      negmatrix = np.clip(matrix,-BIGVALUE,0.0)
      negmatrix -= delta * np.sign(negmatrix)
      np.clip(negmatrix,-BIGVALUE,0.0,out=negmatrix)
      np.add(negmatrix,posmatrix,out=matrix)
      if wtmodel == POS: matrix += self.beta
      
    # NOTE: now do this in caller
    # if truncate = 1, clip updated weights to allowed range
      
    #if self.truncate:
    #  if wtmodel == POSNEG:
    #    np.clip(matrix,-self.clip,self.clip,out=matrix)
    #  else:
    #    np.clip(matrix,self.clipmin,self.clipmax,out=matrix)
