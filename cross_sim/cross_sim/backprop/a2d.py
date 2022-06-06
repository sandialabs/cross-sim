#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# analog to digital conversion models, implemented numerically

import numpy as np

STYLES = ("BASIC")
BASIC = (0)

# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# analog to digital conversion

class A2D:
  """
  activation models
  style = BASIC = add noise, then clip/discretize
  """
  
  def __init__(self,**kwargs):
    style = kwargs["style"]
    bits = kwargs["bits"]
    sigma = kwargs["sigma"]
    if "cliplo" in kwargs: cliplo = kwargs["cliplo"]
    else: cliplo = -1.0
    if "cliphi" in kwargs: cliphi = kwargs["cliphi"]
    else: cliphi = -1.0

    self.bits = bits          # discretized levels = 2^(bits+1) - 1
    self.sigma = sigma        # Gaussian sigma on noise relative to 0-1
    self.cliplo = cliplo      # lo value to clip to
    self.cliphi = cliphi      # hi value to clip to
    self.inactive = 0

    if style == "NONE" or (bits == 0 and sigma == 0.0):
      self.inactive = 1
      return
    
    if style not in STYLES: raise ValueError("Unknown style")
    self.style = STYLES.index(style)
    
    if bits < 0: raise ValueError("Bits cannot be < 0")
    if sigma < 0.0: raise ValueError("Sigma cannot be <= 0.0")

    if bits: self.discrete = 1
    else: self.discrete = 0
    if sigma > 0.0: self.noise = 1
    else: self.noise = 0
    
    # error checks

    if self.noise and (self.sigma < 0.0 or self.sigma > 1.0):
      error("Invalid a2d:sigma arg")
    if self.discrete and self.cliplo >= self.cliphi:
      error("Invalid a2d:clip args")

    # pre-compute
    # nlevel is always odd
      
    self.nlevel = int(2**(self.bits+1) - 1)
    self.nbins = self.nlevel - 1
    self.cliprange = self.cliphi - self.cliplo
    
  # apply function
  # add noise, clip, discretize
  # this MODIFIES x
    
  def apply(self,x):
    style = self.style
    if style == BASIC:

      # add noise first, proportional to cliprange
      
      if self.noise:
        nlen = x.shape
        rvec = np.random.randn(nlen[0])   # mean = 0.0, unit sigma
        x += self.sigma*self.cliprange*rvec

      # clip to range
      # shift/scale to 0-1
      # scale to 0-Nlevel
      # round to nearest int, puts 1/2 bin at 0 and Nlevel
      # unscale/unshift back to clip range
        
      if self.discrete:
        np.clip(x,self.cliplo,self.cliphi,out=x)
        x += self.cliplo
        x *= self.nbins / self.cliprange
        np.rint(x,out=x)     # nearest integer
        x *= self.cliprange / self.nbins
        x -= self.cliplo
