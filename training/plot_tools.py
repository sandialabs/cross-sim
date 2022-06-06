#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# plot_tools module
# Steve Plimpton, Sandia National Labs, sjplimp@sandia.gov
# refomatted for python3
# usage:
#   import plot_tools as PT
#   p = PT.Plot()

# collection of classes and functions for
#   extracting data from files, creating plots

# classes:
#   Extract = extract info from log files into data file
#   Plot = wrapper on matplotlib

# functions:
#   extract_column = extract column of values from table in file

# support modules

import sys,os,shutil,glob,re,time,subprocess,types,string

BIG = 1.0e20

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

# Extract tool
# extract data from files, organize, write to new files for post-analysis

class Extract:
  def __init__(self):
    self.files = []              # used by grep()
    self.otherfiles = []         # used by grep()
    self.search = ""             # used by grep()
    self.nrow = self.ncol = 0    # number of rows and cols in data table
    self.table = []              # table of values 
    self.ops = []                # list of operations on file to populate table
                                 # see op_add()

  # invoke - not documented, not sure how used
  # expand filenames in dir via glob, so can use wildcards
  # search each file for matching string via grep and print results
  # if field is 0, print entire matching line
  # if files is not 0, print only that field from matching line

  def grep(self,field=0):
    if not self.files: error("Extract tool invoke: no files")
    if not self.search: error("Extract tool invoke: no search string")

    self.files.sort()
    self.otherfiles.sort()
    for i,file in enumerate(self.files):
      cmd = 'grep "%s" %s' % (self.search,file)
      out = subprocess.check_output(cmd,shell=True)
      if self.otherfiles:
        cmd = 'grep "%s" %s' % (self.search,self.otherfiles[i])
        otherout = subprocess.check_output(cmd,shell=True)
      if not field:
        print("%s: %s" % (file,out.strip()))
        if self.otherfiles:
          print("%s: %s" % (self.otherfiles[i],otherout.strip()))
      else:
        words = out.split()
        if self.otherfiles: owords = otherout.split()
        if not self.otherfiles: print("%s: %s" % (file,words[field-1]))
        else: print("%s: %s %s" % (file,words[field-1],owords[field-1]))

  # grep a file for regex pattern
  # return list of matching lines
  
  def grep(self,file,pattern):
    if not os.path.isfile(file): error("Extract tool grep: no file")
    lines = open(file,'r').readlines()
    matchlines = []
    for line in lines:
      if re.search(pattern,line): matchlines.append(line)
    return matchlines

  # create a new Nrow by Ncol table, initialize with zeroes
  
  def table_create(self,nrow,ncol):
    self.nrow = nrow
    self.ncol = ncol
    self.table = []
    for i in range(nrow):
      self.table.append(ncol*[0])

  # set element, or entire row or column, or entire table with value
  # row,col > 0 -> one element, indices are 1 to N,M, value = scalar
  # row > 0, col = 0 -> one row, row index is 1 to N, value = list
  # row = 0, col > 0 -> one col, col index is 1 to M, value = list
  # row,col = 0 -> entire table, value = scalar
  
  def table_set(self,row=0,col=0,value=None):
    if value == None: error("Extract tool table_set: requires value")
    if row and col:
      if type(value) == list:
        error("Extract tool table_set: value cannot be list")
      self.table[row-1][col-1] = value
    elif row:
      if type(value) != list:
        error("Extract tool table_set: value must be list")
      for j in range(self.ncol):
        self.table[row-1][j] = value[j]
    elif col:
      if type(value) != list:
        error("Extract tool table_set: value must be list")
      for i in range(self.nrow):
        self.table[i][col-1] = value[i]
    else:
      if type(value) == list:
        error("Extract tool table_set: value cannot be list")
      for i in range(self.nrow):
        for j in range(self.ncol):
          self.table[i][j] = value

# read all tables from filelist and return those with titles matching dict
# filelist can be single filename string containing one or more tables
# filelist can be list of filenames, each with one or more tables
# dict = key/value pairs that must appear in title as key=value
# return list of titles and tables
# returned tables are converted to numeric list of lists
          
  def table_read_all(self,filelist,dict):
    if type(filelist) is str: filelist = [filelist]
    
    titles = []
    tables = []
    for file in filelist:
      txt = open(file,"r").read()
      pattern = re.compile("(^|\n)(#.*?#)\n(.*?)\n(\n|$)",re.DOTALL)
      matches = re.findall(pattern,txt)
      for match in matches:
        title = match[1]
        validmatch = True
        for key,value in list(dict.items()):
          if extract_keyword(title,key) != value: validmatch = False
        if not validmatch: continue
        titles.append(title)
      
        tablelines = match[2].strip().split("\n")
        nrow = len(tablelines)
        ncol = len(tablelines[-1].split())
        table = []
        for i in range(nrow):
          table.append(ncol*[0])
        for i,line in enumerate(tablelines):
          words = line.split()
          for j,word in enumerate(words):
            try: table[i][j] = float(word)
            except: table[i][j] = 0.0
        tables.append(table)

    return titles,tables
      
  # read table from file with title and return it as list of lists
  # title is enclosed by '#' chars
          
  def table_read(self,file,title):
    txt = open(file,"r").read()
    pattern = "(^|\n)#%s#\n(.*?)\n(\n|$)" % title
    match = re.search(pattern,txt,re.DOTALL)
    if not match:
      error("Extract tool table_read:: no table %s in %s" % (title,file))
    table = match.group(2)
    lines = table.split('\n')

    self.nrow = len(lines) - 1
    self.ncol = len(lines[-1].split())
    self.table = []
    for i in range(self.nrow):
      self.table.append(self.ncol*[0])

    for i,line in enumerate(lines[1:]):
      words = line.split()
      for j,word in enumerate(words):
        try: self.table[i][j] = float(word)
        except: self.table[i][j] = 0.0
      
    return self.table

  # count the number of tables in file and return count
  # table titles are enclosed by '#' chars
          
  def table_count(self,file):
    txt = open(file,"r").readlines()
    count = 0
    for line in txt:
      if line[0] == "#":
        line = line.rstrip()
        if line[-1] == "#": count += 1
    return count
  
  # read Nth table from file and return it as list of lists
  # table titles are enclosed by '#' chars
  # returned title has no '#' chars
          
  def table_read_nth(self,file,nth):
    txt = open(file,"r").readlines()
    found = 0
    count = 0
    for i,line in enumerate(txt):
      if line[0] == "#":
        line = line.rstrip()
        if line[-1] == "#":
          count += 1
          if count == nth:
            for j in range(i+2,len(txt)):
              line = txt[j].strip()
              if not line: break
            if j == len(txt)-1: j += 1  # read to end of file w/out blank line
            title = txt[i].strip()
            title = title[1:-1]         # strip leading/trailing '#' char
            lines = txt[i+2:j]
            found = 1
            break
          
    if not found:
      error("Extract tool table_read_nth:: no table %d in %s" % (nth,file))

    self.nrow = len(lines)
    self.ncol = len(lines[-1].split())
    self.table = []
    for i in range(self.nrow):
      self.table.append(self.ncol*[0])

    for i,line in enumerate(lines):
      words = line.split()
      for j,word in enumerate(words):
        try: self.table[i][j] = float(word)
        except: self.table[i][j] = 0.0
      
    return title,self.table
  
  # write table to file, with optional title and column-header lines
  # mode = append,replace
          
  def table_write(self,file,title="",cols="",mode="append"):
    if mode == "append":
      fp = open(file,"a")
      print(file=fp)
      if title: print(title, file=fp)
      if cols: print(cols, file=fp)
      for i in range(self.nrow):
        for j in range(self.ncol):
          print(self.table[i][j], end=' ', file=fp)
        print(file=fp)
      fp.close()
    else:
      # use table_read with flag to see if table exists
      # if so, can use those lines to split file into pre,table,post
      # then write file back out with pre,new-table,post
      pass
  
  # print table to screen, with optional title and column-header lines
  
  def table_print(self,title="",cols=""):
    if title: print(title)
    if cols: print(cols)
    for i in range(self.nrow):
      for j in range(self.ncol):
        print(self.table[i][j], end=' ')
      print()

  # clear list of operations
  
  def op_clear(self):
    self.ops = []

  # append an operation as list to self.ops list
  # "grep" searchstr
  # "index" I J

  def op_add(self,*args):
    keyword = args[0]
    if keyword == "grep":
      if len(args) != 2: error("Extract tool op_add: grep takes 1 arg")
      self.ops.append([keyword,args[1]])
    elif keyword == "index":
      if len(args) != 3: error("Extract tool op_add: index takes 2 args")
      self.ops.append([keyword,int(args[1]),int(args[2])])
    else: error("Extract tool op_add: unrecognized keyword")

  # perfom list of operations on file to extract/return a value
  # file initially read as list of lines
  # "grep" searches file for matching grep str, lines = matching lines
  # "index" extracts Jth value of Ith line, I/J = 1 to N/M
  # return value, which can be used to set table
  
  def file_operate(self,file):
    if not os.path.isfile(file): return "--"
    lines = open(file,'r').readlines()
    for op in self.ops:
      if op[0] == "grep":
        newlines = []
        for line in lines:
          if re.search(op[1],line): newlines.append(line)
        lines = newlines
      if op[0] == "index":
        row = op[1]
        col = op[2]
        if row > len(lines): value = "--"
        else:
          line = lines[row-1]
          words = line.split()
          if col > len(words): value = "--"
          else: value = words[col-1]
    return value

# ----------------------------------------------------------------------

# Plot tool
# simple wrapper on matplotlib

class Plot:
  def __init__(self):
    self.curves = []      # list of entries = [xvec,yvec,flags], see add()
    self.mode = "plot"    # plot, loglog, semilogx, semilogy
    self.aspect = ()      # aspect ratio (width,height)
    self.title = ""       # main title
    self.xtitle = "X"     # X-axis title
    self.ytitle = "Y"     # Y-axis title
    self.aspect = ""      # "" = default, else 1.0, 2.0, 0.5, etc
    self.xlim = ()        # (xlo,xhi), def = adapt to curve data
    self.ylim = ()        # (ylo,yhi), def = adapt to curve data
    self.legend = []      # list of strings, one per curve
    self.legtitle = ""    # legend title
    self.legframe = False # legend frame = True/False
    self.leganchor = ()   # legend anchor pt = (a,b), typically 0 to 1
    self.legcol = 1       # number of columns in legend
    self.legloc = "best"  # legend location
                          # this loc in legend goes at leganchor pt
                          #   best (default),
                          #   left, center, right,
                          #   lower left, lower center, lower right,
                          #   upper left, upper center, upper right,
                          #   center left, center right
    self.font = 16        # font size
    self.linewidth = 2    # line width
    self.markersize = 10  # size of curve marker pts
    self.xtics = []       # list of x tic locations
    self.xlabels = []     # list of x tic labels
    self.ytics = []       # list of y tic locations
    self.ylabels = []     # list of y tic labels
    self.xticoff = 0      # 1 to turn off minor X axis tics
    self.yticoff = 0      # 1 to turn off minor Y axis tics
    self.text = []        # list of entries = [x,y,text,...], see addtxt()

  # add a curve to self.curves
  # x,y = X and Y vectors
  # format = "CML" or "CLM", where M,L can be in either order
  # C = color = r, g, b, y, m (magenta), c (cyan), k (black), w (white)
  # M = marker (optional), no marker if not specified
  #     '.' = point marker
  #     ',' = pixel marker
  #     'o' = circle marker
  #     'v' = triangle_down marker
  #     '^' = triangle_up marker
  #     '<' = triangle_left marker
  #     '>' = triangle_right marker
  #     '1' = tri_down marker
  #     '2' = tri_up marker
  #     '3' = tri_left marker
  #     '4' = tri_right marker
  #     's' = square marker
  #     'p' = pentagon marker
  #     '*' = star marker
  #     'h' = hexagon1 marker
  #     'H' = hexagon2 marker
  #     '+' = plus marker
  #     'x' = x marker
  #     'D' = diamond marker
  #     'd' = thin_diamond marker
  #     '|' = vline marker
  #     '_' = hline marker
  # L = linestyle (optional), solid line if not specified
  #     '-' = line
  #     '--' = dashed line
  #     '-.' = dash/dot line
  #     ':'  = hashed line
  #     ' '  = no line (just markers)
  # face and edge can also be colors (abbrev or name), default to color C
  #     face = white for open symbol

  def add(self,x,y,format,face="",edge=""):
    if not face: face = format[0]
    if not edge: edge = format[0]
    self.curves.append([x,y,format,face,edge])

  # add an annotation text string at x,y in plot coords
  # halign = center (def), right, left
  # valign = center (def), top, bottom, baseline
  # color = text color (default = black)
  # size = font size (default 0.0 = self.font)
    
  def addtxt(self,x,y,text,halign="center",valign="center",color="",size=0.0):
    if not color: color = "black"
    if not size: size = self.font
    self.text.append([x,y,text,halign,valign,color,size])

  # create plot, out = for *.jpg, *.pdf, else just show on screen
  # hook = 1 if want to return plt,ax so caller can invoke more commands
  
  def invoke(self,out="",hook=0):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams['font.size'] = self.font
    mpl.rcParams['lines.linewidth'] = self.linewidth
    mpl.rcParams['lines.markersize'] = self.markersize

    if not self.aspect: fig = plt.figure()
    else: fig = plt.figure(figsize=self.aspect)
    ax = fig.add_subplot(111)
    
    plt.title(self.title)
    plt.xlabel(self.xtitle)
    plt.ylabel(self.ytitle)
    if self.xlim: plt.xlim(self.xlim[0],self.xlim[1])
    if self.ylim: plt.ylim(self.ylim[0],self.ylim[1])

    for i,curve in enumerate(self.curves):
      if self.mode == "plot":
        plt.plot(curve[0],curve[1],curve[2],
                 markerfacecolor=curve[3],markeredgecolor=curve[4])
      if self.mode == "loglog":
        plt.loglog(curve[0],curve[1],curve[2],
                   markerfacecolor=curve[3],markeredgecolor=curve[4])
      if self.mode == "semilogx":
        plt.semilogx(curve[0],curve[1],curve[2],
                     markerfacecolor=curve[3],markeredgecolor=curve[4])
      if self.mode == "semilogy":
        plt.semilogy(curve[0],curve[1],curve[2],
                     markerfacecolor=curve[3],markeredgecolor=curve[4])

    # axis tics
        
    if self.xtics:
      if not self.xlabels: plt.xticks(self.xtics)
      else: plt.xticks(self.xtics,self.xlabels)
    if self.ytics:
      if not self.ylabels: plt.yticks(self.ytics)
      else: plt.yticks(self.ytics,self.ylabels)

    if self.xticoff:
      ax.tick_params(axis='x',which='minor',bottom='off',top='off')
    if self.yticoff:
      ax.tick_params(axis='y',which='minor',left='off',right='off')
    
    # legend
    # don't yet know how to prevent some curves from showing up in legend
    # don't yet know how to get legend to appear when outside of box
    #   either on screen or PDF
      
    lgd = None
    if self.legend:
      if not self.leganchor:
        lgd = ax.legend(self.legend,loc=self.legloc,ncol=self.legcol,
                        title=self.legtitle,frameon=self.legframe)
      else:
        lgd = ax.legend(self.legend,loc=self.legloc,ncol=self.legcol,
                        title=self.legtitle,frameon=self.legframe,
                        bbox_to_anchor=self.leganchor)

    # text labels
      
    if self.text:
      for text in self.text:
        ax.text(text[0],text[1],text[2],
                horizontalalignment=text[3],verticalalignment=text[4],
                color=text[5],fontsize=text[6])

    plt.show()

    if ".jpg" in out or ".pdf" in out:
      if lgd: 
        plt.savefig(out,bbox_extra_artists=(lgd,),bbox_inches="tight")
      else: 
        plt.savefig(out,bbox_inches="tight")

    if hook: return plt,ax
    
# ----------------------------------------------------------------------
# more methods that depend on classes
# ----------------------------------------------------------------------

# extract column from table in file
# return as vector of floating point values
# file = filename
# tname = name of table
# table format in file:
#   #name#" preceeds table
#   1st row is skipped, column headings or comment
#   remaining rows = rows of numbers in columns
#   table ends with 2 blank lines
# icol = which column, 1 to N

def extract_column(file,title,icol):
  e = Extract()
  table = e.table_read(file,title)
  if icol > e.ncol:
    error("Extract_column: table does not have %d columns" % icol)
  vector = e.nrow*[0]
  for i in range(e.nrow): vector[i] = table[i][icol-1]
  return vector

# extract column from table

def extract_column_table(table,icol):
  vector = len(table)*[0]
  for i,row in enumerate(table):
    vector[i] = row[icol-1]
  return vector

# return max value from Ncol column of table, Ncol = 1 to N

def extract_max(table,ncol):
  maxvalue = -BIG
  for i in range(len(table)):
    maxvalue = max(maxvalue,table[i][ncol-1])
  return maxvalue

# extract value associated with keyword from a title string
# words in title string are of form key=value
# strip leading/training "#" char if exists
# return value or None if keyword does not exist
                   
def extract_keyword(title,keyword):
  if title[0] == "#": title = title[1:]
  if title[-1] == "#": title = title[:-1]
  words = title.split()
  for word in words:
    if "=" not in word: continue
    fields = word.split("=")
    if fields[0] == keyword: return fields[1]
  return None

# sort 2 lists together
# sort of first list also orders the second list
# zip produces sequences, convert back into lists

def two_sort(list1,list2):
  list1,list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
  return list1,list2

def two_sort_numpy(list1,list2):
  idx = np.argsort(list1)
  list1 = np.array(list1)[idx]
  list2 = np.array(list2)[idx]
  return list1,list2
