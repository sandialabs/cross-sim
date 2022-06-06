#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Nonlinear models, implemented numerically

import numpy as np
from math import exp

STYLES = ("ASYM","ASYMLOOP","SYM")
ASYM,ASYMLOOP,SYM = (0,1,2)
WTMODELS = ("POSNEG","POS")
POSNEG,POS = (0,1)

# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# numeric Nonlinear models

class Nonlinear:
  """
  nonlinear models
  required args:
    style = ASYM,ASYMLOOP,SYM
    alpha = degree of nonlinearity, smaller is more linear, 0.0 = linear
    clip = +/- clip window on weights for POSNEG wtmodel
    clipmin = min weight value for POS wtmodel
    clipmax = max weight value for POS wtmodel
  optional args:
    wtmodel = POSNEG,POS (def = POSNEG)
      POSNEG = weights are postive and negative
      POS = weights are only positive

  wmin,wmax = lo/hi clip values for either wtmodel
  w0 = (wmax-wmin) / (1 - exp(-alpha))
  deltanorm = delta / (2*wmax)
  w = current weight value
  
  ASYM and ASYMLOOP models = asymmetric nonlinearity
    for delta > 0: NL delta = (w0 + wmin - w) * (1 - exp(-alpha*deltanorm))
    for delta < 0: NL delta: (wmax - w0 - w) * (1 - exp(+alpha*deltanorm))
  SYM model = symmetric nonlinearity
    A = (wmax - wmin) * (exp(alpha)+1) / (exp(alpha)-1)
    B = wmin - (wmax-wmin) / (exp(alpha)-1)
    denominator = 1 + exp(-2*alpha*deltanorm) * (A/(w-B) - 1)
    NL delta = (A / denominator) - w + B
  """
  
  def __init__(self,**kwargs):
    if "style" not in kwargs: error("Style setting required")
    if "alpha" not in kwargs: error("Alpha setting required")
    if "clip" not in kwargs: error("Clip setting required")
    if "clipmin" not in kwargs: error("Clipmin setting required")
    if "clipmax" not in kwargs: error("Clipmax setting required")
    self.style = kwargs["style"]
    self.alpha = kwargs["alpha"]
    self.clip = kwargs["clip"]
    self.clipmin = kwargs["clipmin"]
    self.clipmax = kwargs["clipmax"]

    # optional args

    self.wtmodel = "POSNEG"
    if "wtmodel" in kwargs: self.wtmodel = kwargs["wtmodel"]

    self.inactive = 0
    if self.style == "NONE" or self.alpha == 0.0:
      self.inactive = 1
      return

    # error checks

    if self.style not in STYLES: raise ValueError("Unknown style")
    self.style = STYLES.index(self.style)
    if self.wtmodel not in WTMODELS: error("Unknown wtmodel")
    self.wtmodel = WTMODELS.index(self.wtmodel)

    if self.alpha < 0.0: error("Alpha cannot be < 0.0")
    if self.clip <= 0.0: error("Clip cannot be <= 0.0")
    if self.clipmin <= 0.0: error("Clipmin cannot be <= 0.0")
    if self.clipmax > 1.0: error("Clipmax cannot be > 1.0")
    if self.clipmin >= self.clipmax: error("Clipmin must be < clipmax")

    # nonlinear factors

    if self.wtmodel == POSNEG:
      self.wmin = -self.clip
      self.wmax = self.clip
      self.wmid = 0.0
    elif self.wtmodel == POS:
      self.wmin = self.clipmin
      self.wmax = self.clipmax
      self.wmid = 0.5 * (self.clipmin + self.clipmax)

    w0 = (self.wmax - self.wmin) / (1.0 - exp(-self.alpha))
    self.wpospre = w0 + self.wmin
    self.wnegpre = self.wmax - w0
    self.A = (self.wmax-self.wmin) * \
        (exp(self.alpha)+1.0) / (exp(self.alpha)-1.0)
    self.B = self.wmin - (self.wmax-self.wmin) / (exp(self.alpha)-1.0)

  # apply nonlinearity to delta, which caller will add to matrix of weights
  # does NOT modify matrix, does MODIFY delta
    
  def apply(self,matrix,delta):
    nrow,ncol = matrix.shape
    style = self.style
    alpha = self.alpha
    wmin = self.wmin
    wmax = self.wmax
    wmid = self.wmid
    
    wpospre = self.wpospre
    wnegpre = self.wnegpre
    A = self.A
    B = self.B

    # asymmetric model
    
    # update when weight change is POSITIVE
    # deltanorm = delta/(2.0 * wmax)
    # tmp2 = 1.0 - math.exp(-deltanorm*alpha)
    # newdelta = (w0 + wmin - matrix) * tmp2
    
    # update when weight change is NEGATIVE
    # deltanorm = delta/(2.0 * wmax)
    # tmp3 = 1.0 - math.exp(-deltanorm*alpha)
    # newdelta = (wmax - w0 - matrix) * tmp3
    
    if style == ASYM:
      deltanorm = delta / (wmax-wmin) 
      
      #tmp2 = 1.0 - exp(-deltanorm*alpha)
      tmp2 = -deltanorm
      tmp2 *= alpha
      tmp2 = np.exp(tmp2,out=tmp2)
      tmp2 = 1.0 - tmp2 
        
      #tmp3 = 1.0 - exp(deltanorm*alpha)
      tmp3 = 1.0*deltanorm
      tmp3 *= alpha
      tmp3 = np.exp(tmp3,out=tmp3)
      tmp3 = 1.0 - tmp3 
        
      #if delta[i][j] > 0.0: newdelta = (wpos - matrix[i][j]) * tmp2
      deltapos = delta > 0.0              # index all deltas > 0.0
      matrixpos = matrix[deltapos]        # grab wts where delta > 0.0
      newdeltapos = wpospre - matrixpos   # (w0 - woffset) - orig_wt *
      newdeltapos *= tmp2[deltapos]       #   (1-e^(-delta/A))

      #if delta[i][j] <= 0.0: newdelta = (wneg - matrix[i][j]) * tmp3
      deltaneg = delta <= 0.0             # index all deltas <= 0.0
      matrixneg = matrix[deltaneg]        # grab wts where delta <= 0.0
      newdeltaneg = wnegpre - matrixneg   # (w0 - woffset + 2wmax) - orig_wt *
      newdeltaneg *= tmp3[deltaneg]       #   (1-e^(-delta/A))

      delta[deltapos] = newdeltapos       # reset positive deltas
      delta[deltaneg] = newdeltaneg       # reset negative deltas
            
    # double loop version of asymmetric model (slow)

    elif style == ASYMLOOP:
      for i in range(nrow):
        for j in range(ncol):
          deltanorm = delta[i][j] / (wmax-wmin)
          if delta[i][j] > 0.0:
            tmp2 = 1.0 - exp(-deltanorm*alpha)
            newdelta = (wpospre - matrix[i][j]) * tmp2
          else:
            tmp2 = 1.0 - exp(deltanorm*alpha)
            newdelta = (wnegpre - matrix[i][j]) * tmp2
          delta[i][j] = newdelta

    # symmetric model
    # could rewrite newdelta eq as more fundamental Numpy ops and test speed
    
    elif style == SYM:
      deltanorm = delta / (wmax-wmin) 
      newdelta = A / (1 + np.exp(-2.0*alpha*deltanorm) *
                      (A/(matrix-B) - 1.0)) - matrix + B
      np.copyto(delta,newdelta)
