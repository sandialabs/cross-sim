#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Scripting, task execution, post-analysis tools
# Useful features:
#   invoke list of shell scripts in serial or in parallel
#   perform tasks on local or remote machine
#   for regression test, compare new file to most recent old file
#   substitute for $(foo) format variables in list of commands
#   expand $(foo) format variables into list of commands for benchmarking

import sys,os,shutil,glob,re,time,subprocess,types,string

# ----------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------

# print error string and quit

def RT_error(txt):
    print("ERROR:",txt)
    sys.exit()

# print warning string

def RT_warn(txt):
    print("WARN:",txt)

# expand to full path name, processing ~ and relative path

def RT_fullpath(path):
    return os.path.abspath(os.path.expanduser(path))

# ----------------------------------------------------------------------

# Run tool

class RT_Run:
    def __init__(self):
        self.file = ""
        self.filelist = []
        self.shell = "sh"

    # one action, use self.file
    # cd into dir, run single file with its args via self.shell
    # if out = 1 (default) print output to screen, else suppress

    def one(self,out=1):
        if not self.file: RT_error("No file in Run tool")
        tstart = time.time()
        path = RT_fullpath(self.file)
        cmd = "cd %s; %s %s" % \
            (os.path.dirname(path),self.shell,os.path.basename(path))
        if out: subprocess.call(cmd,shell=True)
        else:
            FNULL = open(os.devnull,'w')
            subprocess.call(cmd,stdout=FNULL,shell=True)
        tstop = time.time()
        print("Total run time for one file = %g secs" % (tstop-tstart))

    # loop action, use self.filelist, one at a time
    # for each, cd into dir, run file with its args via self.shell
    # if out = 1 (default) print output to screen, else suppress

    def loop(self,out=1):
        if not self.filelist: RT_error("No filelist in Run tool")
        tstart = time.time()
        for i,file in enumerate(self.filelist):
            print("launching task",i+1)
            path = RT_fullpath(file)
            cmd = "cd %s; %s %s" % \
                (os.path.dirname(path),self.shell,os.path.basename(path))
            if out: subprocess.call(cmd,shell=True)
            else:
                FNULL = open(os.devnull,'w')
                subprocess.call(cmd,stdout=FNULL,shell=True)
        tstop = time.time()
        print("Total run time for loop = %g secs" % (tstop-tstart))

    # parallel action:
    # treat list of files to invoke as parallel tasks
    # run in master/slave fashion with up to Nprocs processes
    # for each task, cd into dir, run file
    # FNULL used to turn off multiple commands writing to stdout

    def parallel(self,nprocs,delay=1.0):
        if not self.filelist: RT_error("No filelist in Run tool")
        ntasks = len(self.filelist)
        processes = nprocs*[-1]
        tstart = time.time()

        ncomplete = nrunning = nlaunched = 0
        nleft = ntasks

        while ncomplete < ntasks:
            if nleft and nrunning < nprocs:
                dir = os.path.dirname(self.filelist[nlaunched])
                file = os.path.basename(self.filelist[nlaunched])
                cmd = "cd %s; %s %s" % (dir,self.shell,file)
                FNULL = open(os.devnull,'w')
                processes[nrunning] = subprocess.Popen(cmd,stdout=FNULL,shell=True)
                nlaunched += 1
                nrunning += 1
                nleft -= 1
                print("launched task",nlaunched)
                continue
            for i,process in enumerate(processes):
                if process == -1: continue
                status = process.poll()
                if status != None:
                    ncomplete += 1
                    print("completed task",ncomplete)
                    if nleft:
                        dir = os.path.dirname(self.filelist[nlaunched])
                        file = os.path.basename(self.filelist[nlaunched])
                        cmd = "cd %s; %s %s" % (dir,self.shell,file)
                        FNULL = open(os.devnull,'w')
                        processes[i] = subprocess.Popen(cmd,stdout=FNULL,shell=True)
                        nlaunched += 1
                        nleft -= 1
                        print("launched task",nlaunched)
                    else:
                        processes[i] = -1
                        nrunning -= 1
            time.sleep(delay)

        tstop = time.time()
        print("Total run time for parallel loop = %g secs" % (tstop-tstart))
