#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# sim_tools module
# Steve Plimpton, Sandia National Labs, sjplimp@sandia.gov
# usage:
#   import sim_tools as ST
#   title = ST.Title()

# collection of classes and functions for helping with HAANA simulations

# classes:
#   Permute = generate permutations from one or more lists
#   Title = generate titles for chunks of output data

# support modules

import sys

# ----------------------------------------------------------------------
# helper functions, called by other methods in this file
# ----------------------------------------------------------------------

# print error string and quit

def error(txt):
  print("ERROR:",txt)
  sys.exit()

# ----------------------------------------------------------------------
# classes
# ----------------------------------------------------------------------

# Permute tool
# generate permutations from a series of one or more lists

class Permute:
  def __init__(self,*args):
    self.params = list(args)
    self.npermute = 1
    for param in self.params: self.npermute *= len(param)
    
  # return the Ith permutation selected by index = 0 to N-1
  # last param varies fastest, first param varies slowest
    
  def one(self,index):
    indices = []
    for i in range(len(self.params)-1,-1,-1):
      param = self.params[i]
      remainder = index % len(param)
      indices.insert(0,remainder)
      index //= len(param)          # index = index // len, "//" = int divide
    params = [self.params[i][indices[i]] for i in range(len(self.params))]
    return params
  
# ----------------------------------------------------------------------

# Title tool
# generate titles from keywords

class Title:
  def __init__(self,prefix="",keywords=[],valtypes=[]):
    self.prefix = prefix
    self.keywords = keywords
    self.valtypes = valtypes

    if len(self.keywords) != len(self.valtypes):
      error("Title: keywords and valtypes are not same length")

  def generate(self,params):
    if len(params) != len(self.keywords):
      error("Title: keywords and params are not same length")
    title = self.prefix
    for i,value in enumerate(params):
      title += " %s=%s" % (self.keywords[i],self.valtypes[i])
      title = title % value
    return title
