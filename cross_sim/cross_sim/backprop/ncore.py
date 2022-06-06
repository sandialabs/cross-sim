#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Neural core, implemented numerically

import numpy as np

from .readnoise import ReadNoise
from .writenoise import WriteNoise
from .nonlinear import Nonlinear
from .a2d import A2D
from .d2a import D2A

WTMODELS = ("POSNEG","POS")
POSNEG,POS = (0,1)

# error output

def error(str, *args, **kwargs): raise ValueError(str, *args, **kwargs)

# Ncore class

class Ncore:
  """
  This class models a single crossbar numerically.  It stores
  resistances/conductances corresponding to matrix elements.
  It stores several noise and precision models for reading and
  writing the weights.  It has 3 kernel methods invoked on
  the crossbar by calling algorithms:
    matvec multiply, vecmat multiply, outer-product update

  Internal data:
  nrow,ncol = size of matrix
  matrix = matrix weights
  mcopy = scratch space of same size as matrix
  """
    
  def __init__(self,nrow,ncol,precision=np.float32,**kwargs):
    """
    define a matrix and how its weights are stored
    required args:
      nrow,ncol = size of matrix
    optional args:
      precision = precision of Numpy matrices, vectors (def = float32)
      wtmodel = POSNEG or POS (def = POSNEG)
        POSNEG = store weights as-is, as positive/negative values
        POS = store weights as positive values from clipmin to clipmax
      clip = +/- clip window on weights for POSNEG wtmodel (def = 1.0)
      clipmin = min weight value for POS wtmodel (def = 0.1)
      clipmax = max weight value for POS wtmodel (def = 1.0)
      truncate = 1/0 for whether to clip wts at range boundaries (def = 1)
        if yes, happens after readnoise is applied and at end of training
        also happens when matrices are first stored
    internal data:
      readnoise = read noise model
      writenoise = write noise model
      nonlinear = nonlinear model
      a2d,d2a = A/D and D/A models for noise and precision
    """

    # allocate matrix with Numpy precision
    
    self.nrow = nrow
    self.ncol = ncol

    # optional args
    
    self.wtmodel = "POSNEG"
    self.clip = 1.0
    self.clipmin = 0.1
    self.clipmax = 1.0
    self.truncate = 1
    self.useGPU = False
    if "wtmodel" in kwargs: self.wtmodel = kwargs["wtmodel"]
    if "clip" in kwargs: self.clip = kwargs["clip"]
    if "clipmin" in kwargs: self.clipmin = kwargs["clipmin"]
    if "clipmax" in kwargs: self.clipmax = kwargs["clipmax"]
    if "truncate" in kwargs: self.truncate = kwargs["truncate"]
    if "useGPU" in kwargs: self.useGPU = kwargs["useGPU"]

    # error check
    
    if self.wtmodel not in WTMODELS: error("Unknown wtmodel")
    self.wtmodel = WTMODELS.index(self.wtmodel)

    if self.clip <= 0.0: error("Clip cannot be <= 0.0")
    if self.clipmin <= 0.0: error("Clipmin cannot be <= 0.0")
    if self.clipmax > 1.0: error("Clipmax cannot be > 1.0")
    if self.clipmin >= self.clipmax: error("Clipmin must be < clipmax")
    if self.truncate not in (0,1): error("Truncate must be 0/1")

    # setup weight models

    self.alpha = (self.clipmax-self.clipmin) / (2.0*self.clip)
    self.invalpha = 1.0/self.alpha
    self.beta = 0.5 * (self.clipmin + self.clipmax)

    self.mv_betavec = np.zeros((self.ncol,1),precision)
    self.mv_betacopy = np.zeros((self.ncol,1),precision)
    self.mv_betavec = self.beta

    self.vm_betavec = np.zeros((self.nrow,1),precision)
    self.vm_betacopy = np.zeros((self.nrow,1),precision)
    self.vm_betavec = self.beta

    # no noise/precision models by default
    
    self.readnoise = None
    self.writenoise = None
    self.nonlinear = None
    self.a2d = None
    self.d2a = None

    global ncp
    if self.useGPU:
        import cupy as cp
        ncp = cp
    else:
        ncp = np

    self.matrix = ncp.zeros((nrow,ncol),precision)
    self.mcopy = ncp.zeros((nrow,ncol),precision)


  # -------------------------------------------------------
  # Allow caller to set and get matrix
  # caller uses POSNEG values, Ncore may store them differently
  # -------------------------------------------------------
    
  def set_matrix(self,matrix,applyDecay=False,verbose=True):
    """
    store matrix after initialized by caller
    remap weights if required by wtmodel
    """

    if self.useGPU:
      matrix = ncp.array(matrix)

    minval = ncp.min(matrix)
    maxval = ncp.max(matrix)
    if verbose: print("Min/Max matrix values",minval,maxval);

    # copyto() will convert any float to precision of self.matrix

    ncp.copyto(self.matrix,matrix)
    matrix = self.matrix

    # map POSNEG weights to POS weights if needed
    # clip both POSNEG and POS matrix to clip window if truncate enabled

    if self.wtmodel == POSNEG:
      if self.truncate: ncp.clip(matrix,-self.clip,self.clip,out=matrix)
    elif self.wtmodel == POS:
      matrix *= self.alpha
      matrix += self.beta
      if self.truncate:
        ncp.clip(matrix,-self.clipmin,self.clipmax,out=matrix)

  def get_matrix(self):
    """
    return matrix to caller
    unmap weights if required by wtmodel
    """
    
    if self.wtmodel == POSNEG: return self.matrix

    # map POS weights to POSNEG weights

    mcopy = self.mcopy
    ncp.copyto(mcopy,self.matrix)
    mcopy -= self.beta
    mcopy *= self.invalpha
    return mcopy

  # -------------------------------------------------------
  # Define readnoise, writenoise, nonlinear, A2D/D2A models
  # pass additional params for weight model if needed
  # -------------------------------------------------------
    
  def set_readnoise(self,**kwargs):
    kwargs["wtmodel"] = WTMODELS[self.wtmodel]
    kwargs["clip"] = self.clip
    kwargs["clipmin"] = self.clipmin
    kwargs["clipmax"] = self.clipmax
    self.readnoise = ReadNoise(**kwargs)
    if self.readnoise.inactive: self.readnoise = None
    
  def set_writenoise(self,**kwargs):
    kwargs["wtmodel"] = WTMODELS[self.wtmodel]
    kwargs["clip"] = self.clip
    kwargs["clipmin"] = self.clipmin
    kwargs["clipmax"] = self.clipmax
    self.writenoise = WriteNoise(**kwargs)
    if self.writenoise.inactive: self.writenoise = None

  def set_nonlinear(self,**kwargs):
    kwargs["wtmodel"] = WTMODELS[self.wtmodel]
    kwargs["clip"] = self.clip
    kwargs["clipmin"] = self.clipmin
    kwargs["clipmax"] = self.clipmax
    self.nonlinear = Nonlinear(**kwargs)
    if self.nonlinear.inactive: self.nonlinear = None

  def set_a2d(self,**kwargs):
    self.a2d = A2D(**kwargs)
    if self.a2d.inactive: self.a2d = None
    
  def set_d2a(self,**kwargs):
    self.d2a = D2A(**kwargs)
    if self.d2a.inactive: self.d2a = None
    
  # -------------------------------------------------------
  # matrix-vector multiply kernel
  # -------------------------------------------------------
    
  def matvec(self,vec,debug=0):
    """
    perform matrix-vector multiply on analog vector
    apply readnoise model to matrix values
    d2a and a2d conversions on in/out vectors
    does NOT modify input vector
    if weights are POS, adjust result as if weights had been POSNEG
    """

    matrix = self.matrix
    readnoise = self.readnoise
    d2a = self.d2a
    a2d = self.a2d
    v = vec.copy()               # so d2a does not change caller's vector
    
    if d2a: d2a.apply(v)

    # apply readnoise and truncate to clip bounds if requested
    # use mcopy so readnoise does not change caller's matrix
    
    if readnoise:
      mcopy = self.mcopy
      ncp.copyto(mcopy,matrix)
      readnoise.apply(mcopy,debug)
      if self.truncate:
        if self.wtmodel == POSNEG:
          ncp.clip(mcopy,-self.clip,self.clip,out=mcopy)
        elif self.wtmodel == POS:
          ncp.clip(mcopy,self.clipmin,self.clipmax,out=mcopy)
      result = mcopy.dot(v)
    else:
      result = matrix.dot(v)

    # convert POS wt result to POSNEG wt result
    # A = original posneg matrix
    # A = shifted/scaled pos matrix
    # A' = alpha A + beta, alpha = ratio of clipranges, beta = shift
    # A'x = pos vector = (alpha A + beta) x = alpha A x + matrix-of-all-beta x
    # Ax = posneg vector = (A'x - beta (x.sum)) / alpha
    # if readnoise: use beta as vector and apply noise, then dot with v
    # this is b/c shift will be performed in hardware with extra memristor/col
      
    if self.wtmodel == POS:
      if readnoise:
        betacopy = self.mv_betacopy
        ncp.copyto(betacopy,self.mv_betavec)
        readnoise.apply(betacopy,debug)
        if self.truncate:
          ncp.clip(betacopy,self.clipmin,self.clipmax,out=betacopy)
        dot = ncp.dot(v,betacopy[:,0])
        ncp.subtract(result,dot,out=result)
        ncp.multiply(result,self.invalpha,out=result)
      else:
        sum = ncp.sum(v)
        ncp.subtract(result,self.beta*sum,out=result)
        ncp.multiply(result,self.invalpha,out=result)
    
    if a2d: a2d.apply(result)
    return result
  
  # -------------------------------------------------------
  # vector-matrix multiply kernel
  # -------------------------------------------------------

  def vecmat(self,vec,debug=0):
    """
    perform vector-matrix multiply on analog vector
    apply readnoise model to matrix values
    d2a and a2d conversions on in/out vectors
    does NOT modify input vector
    if weights are POS, adjust result as if weights had been POSNEG
    """

    matrix = self.matrix
    readnoise = self.readnoise
    d2a = self.d2a
    a2d = self.a2d
    v = vec.copy()               # so d2a does not change caller's vector

    if d2a: d2a.apply(v)

    # apply readnoise and truncate to clip bounds if requested
    # use mcopy so readnoise does not change caller's matrix

    if readnoise:
      mcopy = self.mcopy
      ncp.copyto(mcopy,matrix)
      readnoise.apply(mcopy,debug)
      if self.truncate:
        if self.wtmodel == POSNEG:
          ncp.clip(mcopy,-self.clip,self.clip,out=mcopy)
        elif self.wtmodel == POS:
          ncp.clip(mcopy,self.clipmin,self.clipmax,out=mcopy)
      result = v.dot(mcopy)
    else:
      result = v.dot(matrix)

    # convert POS wt result to POSNEG wt result
    # A = original posneg matrix
    # A = shifted/scaled pos matrix
    # A' = alpha A + beta, alpha = ratio of clipranges, beta = shift
    # A'x = pos vector = (alpha A + beta) x = alpha A x + matrix-of-all-beta x
    # Ax = posneg vector = (A'x - beta (x.sum)) / alpha
    # if readnoise: convert beta to vector and apply noise before summing
    # this is b/c shift will be performed in hardware with extra memristor/col
      
    if self.wtmodel == POS:
      if readnoise:
        betacopy = self.vm_betacopy
        ncp.copyto(betacopy,self.vm_betavec)
        readnoise.apply(betacopy,debug)
        if self.truncate:
          ncp.clip(betacopy,self.clipmin,self.clipmax,out=betacopy)
        dot = ncp.dot(v,betacopy[:,0])
        ncp.subtract(result,dot,out=result)
        ncp.multiply(result,self.invalpha,out=result)
      else:
        sum = ncp.sum(v)
        ncp.subtract(result,self.beta*sum,out=result)
        ncp.multiply(result,self.invalpha,out=result)
            
    if a2d: a2d.apply(result)
    return result
    
  # -------------------------------------------------------
  # outer-product matrix update kernel
  # -------------------------------------------------------

  def update(self,vec1,vec2,rate,debug=0):
    """
    perform matrix update via outer product of 2 analog vectors
    rate = learning rate
    apply nonlinear and writenoise models to matrix values
    d2a and a2d conversions on 2 input vectors
    does NOT modify input vectors
    if weights are POS, adjust result as if weights had been POSNEG
    """

    matrix = self.matrix
    nonlinear = self.nonlinear
    writenoise = self.writenoise
    d2a = self.d2a
    a2d = self.a2d
    v1 = vec1.copy()             # so d2a does not change caller's vectors
    v2 = vec2.copy()

    if d2a:
      d2a.apply(v1)
      d2a.apply(v2)

    # delta = outer product update to apply to matrix
    # for wtmodel POS, convert delta to matrix scale via ratio of clip ranges
    # adjust delta by nonlinear and write noise terms
    
    delta = rate * ncp.outer(v1,v2)
    
    #if debug:
    #  print("V1",v1)
    #  print("V2",v2)
    #  print("DELTA",delta[0,:])
    
    if self.wtmodel == POS: delta *= self.alpha

    if nonlinear: nonlinear.apply(matrix,delta)
    if writenoise: writenoise.apply(matrix,delta)
    matrix += delta

    # truncate updated weights to clip limits if requested

    if self.truncate:
      if self.wtmodel == POSNEG:
        ncp.clip(matrix,-self.clip,self.clip,out=matrix)
      elif self.wtmodel == POS:
        ncp.clip(matrix,self.clipmin,self.clipmax,out=matrix)


  # -------------------------------------------------------
  # burst matrix update kernel
  # -------------------------------------------------------

  def update_burst(self,update_mat,rate,debug=0):
    """
    perform matrix update using update matrix input "mat"
    rate = learning rate
    apply nonlinear and writenoise models to matrix values
    d2a and a2d conversions on 2 input vectors
    does NOT modify input vectors
    if weights are POS, adjust result as if weights had been POSNEG
    """

    matrix = self.matrix
    nonlinear = self.nonlinear
    writenoise = self.writenoise
    d2a = self.d2a
    a2d = self.a2d
    umat = update_mat.copy()             # so d2a does not change caller's vectors
    
    if d2a:
      d2a.apply(umat)

    # delta = outer product update to apply to matrix
    # for wtmodel POS, convert delta to matrix scale via ratio of clip ranges
    # adjust delta by nonlinear and write noise terms
    
    delta = rate * umat
    
    #if debug:
    #  print("V1",v1)
    #  print("V2",v2)
    #  print("DELTA",delta[0,:])
    
    if self.wtmodel == POS: delta *= self.alpha

    if nonlinear: nonlinear.apply(matrix,delta)
    if writenoise: writenoise.apply(matrix,delta)
    matrix += delta

    # truncate updated weights to clip limits if requested

    if self.truncate:
      if self.wtmodel == POSNEG:
        ncp.clip(matrix,-self.clip,self.clip,out=matrix)
      elif self.wtmodel == POS:
        ncp.clip(matrix,self.clipmin,self.clipmax,out=matrix)

