#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__author__ = 'sagarwa'

import os, re
import numpy as np
from numpy import ndarray
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import LinearNDInterpolator
import scipy as sp
from scipy import stats
from scipy.stats.kde import gaussian_kde
import warnings

#  all data processing functions should return two lists of ndarrays, 1st of increasing pulses, 2nd of decreasing pulses.  The data in each list should be conductance values after a pulse
class process_data():
    def __init__(self, datapath = '', scale=1e6, scale_text="$\mu$", G_ticks=None, outdir=None):
        self.datapath = datapath
        if outdir is None:
            self.outdir = datapath
        else:
            self.outdir=outdir
        self.scale =scale
        self.scale_text=scale_text
        self.G_ticks =G_ticks

    def find_min_max(self,pulses):
        # find minimum conductance
        first_val = True
        for ind in range (len(pulses)):
            if len(pulses[ind])!=0:
                min1 = np.min(pulses[ind])
                max1 = np.max(pulses[ind])
                if first_val:
                    first_val=False
                    min_G = min1
                    max_G = max1
                else:
                    min_G  = min(min_G,min1)
                    max_G = max(max_G,max1)
        return min_G, max_G


    #***************** funtions to load data in different formats *********************************
    def load_set_reset_csv(self, set_file, reset_file, skip_header, skip_footer=0, read_voltage=0.1):
        # ****************  function to process either increasing or decreasing pulses
        def process_data(data):
            n_runs = data.shape[1]

            # convert to list of ndarrays
            processed_data = []
            for ind in range(n_runs):
                # eliminate NaNs and zeros from arrays
                values = data[:,ind]
                values = values[~np.isnan(values)]
                values = values[values!=0]
                values = values/read_voltage  # convert current to conductance
                if len(values)>0: #make sure column is not empty
                    processed_data.append(values)
            return processed_data

        data = np.genfromtxt(os.path.join(self.datapath,set_file), skip_header=skip_header, delimiter=',' ,skip_footer=skip_footer)
        increasing_pulses_proccesed = process_data(data)

        data = np.genfromtxt(os.path.join(self.datapath,reset_file), skip_header=skip_header, delimiter=',' ,skip_footer=skip_footer)
        decreasing_pulses_proccesed = process_data(data)

        return  increasing_pulses_proccesed, decreasing_pulses_proccesed

    def load_single_col_data(self, file, skip_header=1, skip_footer=0, pulses_per_ramp=50,read_voltage=1,current_col=1, includes_starting_pulse=True):
        """
        Load data from a file that has each ramp squentially after the next with a fixed number of pulses in each ramp
        :param file:
        :param skip_header:
        :param skip_footer:
        :param read_voltage:
        :param current_col: col that has current (indexed from 0)
        :param includes_starting_pulse: is an extra measurement to get the starting conductance included?
        :return:
        """
        data = np.genfromtxt(os.path.join(self.datapath,file), skip_header=skip_header, delimiter=',' ,skip_footer=skip_footer,usecols=[current_col])

        if includes_starting_pulse:
            n_ramps = (data.size-1)/pulses_per_ramp/2 # has a starting and an ending conductance read
        else:
            n_ramps = data.size/pulses_per_ramp/2 # has a starting and an ending conductance read

        if not np.isclose(int(n_ramps), n_ramps):
            raise ValueError("The data does not have an even number of equal sized ramps")
        increasing_pulses = []
        decreasing_pulses =[]
        for ind in range (int(n_ramps)):
            if includes_starting_pulse:
                increasing_pulses.append(data[pulses_per_ramp*2*ind:pulses_per_ramp*(2*ind+1)+1]/read_voltage) # add extra overlapping pulse at end to capture start/end values
                decreasing_pulses.append(data[pulses_per_ramp*(2*ind+1):pulses_per_ramp*(2*ind+2)+1]/read_voltage)
            else:
                if ind==0:
                    increasing_pulses.append(data[0:pulses_per_ramp] / read_voltage)
                    decreasing_pulses.append(data[pulses_per_ramp:pulses_per_ramp *2] / read_voltage)
                else:
                    increasing_pulses.append(data[pulses_per_ramp * 2 * ind -1 :pulses_per_ramp * (2 * ind + 1)] / read_voltage)  # add extra overlapping pulse at start to capture start/end values
                    decreasing_pulses.append(data[pulses_per_ramp * (2 * ind + 1)-1:pulses_per_ramp * (2 * ind + 2)] / read_voltage)

        return increasing_pulses,decreasing_pulses


    def process_alberto_data(self, file, header_lines=4, skip_footer=3, read_voltage=0.1):
        data = np.genfromtxt(os.path.join(self.datapath,file), delimiter=',', skip_header=header_lines, skip_footer=skip_footer)
        print(data)
        time_reset = data[:,0]
        G_reset = data[:,1]/-read_voltage
        time_set = data[:,2]
        G_set = data[:,3]/-read_voltage


        decreasing_pulses = []
        increasing_pulses = []

        ind =0
        pulses = np.array([])

        # decreasing reset pulses
        while True:
            if ind==len(time_reset)-1:
                break
            if (time_reset[ind+1]-time_reset[ind])>100:
                decreasing_pulses.append(pulses)
                pulses = np.array([])
            else:
                pulses = np.append(pulses, G_reset[ind])
            ind+=1

        # increasing set pulses
        ind =0
        pulses = np.array([])
        while True:
            if ind==len(time_set)-1:
                break
            if (time_set[ind+1]-time_set[ind])>100:
                increasing_pulses.append(pulses)
                pulses = np.array([])
            else:
                pulses = np.append(pulses, G_set[ind])
            ind+=1


        plt.figure()
        for ind in range(len(increasing_pulses)):
            plt.plot(increasing_pulses[ind]*self.scale,'-o')
        plt.figure()
        for ind in range(len(decreasing_pulses)):
            plt.plot(decreasing_pulses[ind]*self.scale,'-o')

        return  increasing_pulses, decreasing_pulses



    def load_IV_data (self, set_file_text, reset_file_text, skip_header=5, skip_footer=0, read_voltage = 0.1,
                      read_gate_voltage=1,current_col=2, voltage_col=1):
        """
        extract data from IV sweeps after each pulse

        :param set_file_text: text present in the filename of every set file
        :param reset_file_text: text present in the filename of every reset file
        :param read_voltage: what drain voltage used during the IV sweep
        :param current_col: which column of the easy expert file contains the current (indexed from 0)
        :param voltage_col: which column of the easy expert file contains the voltage (indexed from 0)
        :param read_gate_voltage: what read gate voltage to look for in the I-V sweep

        """


        # ****************  function to process either increasing or decreasing pulses
        def process_data(fileprefix):
            files = glob.glob(fileprefix)
            if files==[]:
                raise ValueError("No data files found, file prefix= "+fileprefix)

            # sort filenames for debugging purposes
            def natural_sort(l):
                convert = lambda text: int(text) if text.isdigit() else text.lower()
                alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
                return sorted(l, key = alphanum_key)
            files = natural_sort(files)

            # create list of ndarrays
            processed_data = []

            for file in files:
                data = np.genfromtxt(file, skip_header=skip_header ,skip_footer=skip_footer,delimiter=',')
                V = data[:,voltage_col]
                I = data[:,current_col]
                I = I[np.isclose(V,read_gate_voltage)]
                G = I/read_voltage

                # eliminate NaNs from arrays
                values = G
                values = values[~np.isnan(values)]
                processed_data.append(values)
            return processed_data


        increasing_pulses_proccesed = process_data(self.datapath+'/*/*'+set_file_text+'*')
        decreasing_pulses_proccesed = process_data(self.datapath+'/*/*'+reset_file_text+'*')

        # print(increasing_pulses_proccesed)

        return  increasing_pulses_proccesed, decreasing_pulses_proccesed



    def load_easy_expert_data (self, set_file_prefix, reset_file_prefix, skip_header=2, skip_footer=0, read_voltage = 1, current_col=1):
        """

        :param current_col: which column of the easy expert file contains the current
        """


        # ****************  function to process either increasing or decreasing pulses
        def process_data(fileprefix):
            files = glob.glob(fileprefix)
            if files==[]:
                raise ValueError("No data files found")

            # sort filenames for debugging purposes
            def natural_sort(l):
                convert = lambda text: int(text) if text.isdigit() else text.lower()
                alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
                return sorted(l, key = alphanum_key)
            files = natural_sort(files)


            # create list of ndarrays
            processed_data = []

            for file in files:
                data = np.genfromtxt(file, skip_header=skip_header ,skip_footer=skip_footer, usecols = (0,current_col)) #delimiter=','
                I = data[:,1]
                G = I/read_voltage

                # eliminate NaNs from arrays
                values = G
                values = values[~np.isnan(values)]
                processed_data.append(values)
            return processed_data

        # print(os.path.join(self.datapath,set_file_prefix+'*'))
        increasing_pulses_proccesed = process_data(os.path.join(self.datapath,set_file_prefix+'*'))
        decreasing_pulses_proccesed = process_data(os.path.join(self.datapath,reset_file_prefix+'*'))

        return  increasing_pulses_proccesed, decreasing_pulses_proccesed


    def process_elliot_raw_data(self, file, header_lines=4, skip_footer=0):
        """
        Processes the raw I-V curve.  Finds each rising/falling edge to compensate for stretched pulses

        :return:
        """
        data = np.genfromtxt(file, skip_header=(header_lines+18), skip_footer=skip_footer)
        plt.plot(data)

        decreasing_pulses = []
        increasing_pulses = []
        num_cycles =40
        max_pulses =10000

        # divide data into series of I-V pulse trains
        for ind in range(num_cycles):
            print("index=" +str(ind))
            min_index = np.argmin(data[0:max_pulses])
            decreasing_pulses.append(data[0:min_index])
            data = data[min_index:]

            max_index = np.argmax(data[0:max_pulses])
            increasing_pulses.append(data[0:max_index])
            data = data[max_index:]

        # # data runs 32 and 33 have funny business going on
        # increasing_pulses = [increasing_pulses[32]]
        # decreasing_pulses = decreasing_pulses[26:27]

        plt.figure()
        for ind in range(len(increasing_pulses)):
            plt.plot(increasing_pulses[ind],'-o')
        plt.figure()
        for ind in range(len(decreasing_pulses)):
            plt.plot(decreasing_pulses[ind],'-o')

        # create function to process a pulse train.  Assumes pulse train is starting in a read state and then alternates between reads and writes
        n_starting_pts = 15 # number of points to skip to get to the center of the first read state
        average_pulse_width =22

        def process_pulse_train(sample, increasing):
            sample = sample[n_starting_pts:]

            edge = sample[:average_pulse_width]
            edge_val = (edge.min() + edge.max())/2
            first_edge_index = (np.abs(edge-edge_val)).argmin()
            if increasing:
                first_pulse_limit = edge.max()
            else:
                first_pulse_limit =edge.min()

            values = []
            while True:
                try:
                    # find 3rd edge
                    max_lim = average_pulse_width*3

                    edge = sample[average_pulse_width*2:max_lim]

                    # detect if the pulse was stretched (max value not larger than previous max value for increasing pulses)
                    # if stretched, increase edge window
                    bad_measurement = False
                    if (increasing and edge.max()<first_pulse_limit) or \
                       (not increasing and edge.min()>first_pulse_limit):
                        bad_measurement = True
                        while True:
                            # print(max_lim)

                            max_lim+=1
                            if max_lim>len(sample):
                                break
                            edge = sample[average_pulse_width*2:max_lim]
                            if (increasing and edge.max()>first_pulse_limit) or \
                               (not increasing and edge.min()<first_pulse_limit):
                                break

                    if bad_measurement:
                        max_lim+=int(average_pulse_width/3) # add extra points to get towards center of pulse
                        edge = sample[average_pulse_width*2:max_lim]

                    edge_val = (edge.min() +edge.max())/2
                    third_edge_index = (np.abs(edge-edge_val)).argmin()+average_pulse_width*2


                    # print("first edge index = "+str(first_edge_index))
                    # print("third edge index = "+str(third_edge_index))

                    # find 2nd edge using range in between 1st and 3rd edge
                    edge = sample[int(first_edge_index+average_pulse_width/4):int(third_edge_index-average_pulse_width/4)]
                    edge_val = (edge.min() +edge.max())/2


                    second_edge_index = (np.abs(edge-edge_val)).argmin() + int(first_edge_index+average_pulse_width/4)

                    # reset the first pulse limit to be max/min of current 2nd pulse
                    if increasing:
                        first_pulse_limit = sample[third_edge_index:int(third_edge_index+average_pulse_width/3)].max()
                    else:
                        first_pulse_limit = sample[third_edge_index:int(third_edge_index+average_pulse_width/3)].min()


                    measuring_index = round( (second_edge_index+third_edge_index)/2)
                    values.append( sample[measuring_index] )
                    sample = sample[measuring_index:-1]
                    first_edge_index = third_edge_index-measuring_index


                except:
                    break
            return values

        increasing_pulses_proccesed = []
        for ind in range(len(increasing_pulses)):
            values=process_pulse_train(increasing_pulses[ind], increasing=True)
            increasing_pulses_proccesed.append(np.array(values))

        decreasing_pulses_proccesed = []
        for ind in range(len(decreasing_pulses)):
            values=process_pulse_train(decreasing_pulses[ind], increasing=False)
            decreasing_pulses_proccesed.append(np.array(values))



        plt.figure()
        for ind in range(len(increasing_pulses)):
            plt.plot(increasing_pulses_proccesed[ind],'-o')

        plt.figure()
        for ind in range(len(decreasing_pulses)):
            plt.plot(decreasing_pulses_proccesed[ind],'-o')
        return  increasing_pulses_proccesed, decreasing_pulses_proccesed

    def process_access_device_sweeps(self, file, header_lines, skip_footer=0, num_cycles=1, num_pulses=10000, amp=1,
                                     full_sweep=False, fig_prefix ='', save=False):
        """
        Processes the raw I-V curve of access device.
        Finds increasing and decreasing ramps, ignoring first half increasing ramp
        :param file: file to read IV data from
        :param header_lines: number of lines to ignore at top of file
        :param skip_footer: number of lines to ignore at end of file
        :param num_cycles: number of full cycles in voltage
        :param num_pulses: number of pulses per cycle
        :param amp: amplitude of the sweep, for a positive sweep this is maximum voltage
        :param full_sweep: true if sweep includes both positive and negative voltages, false if only positive

        :return:
        """

        data = np.genfromtxt(os.path.join(self.datapath, file), delimiter=',', skip_header=header_lines,
                             skip_footer=skip_footer)

        ramps = data[1::2]

        if not full_sweep:
            # Function for finding positive and negative turn on voltage
            def find_turning_voltage(ramp_data):
                for ind in range(len(ramp_data)):
                    if ramp_data[ind] > .0000001:
                        return amp * (ind / (.5*num_pulses)), None
                raise ValueError("No turning voltage found.")
        else:
            # Function for finding positive turn on voltage
            def find_turning_voltage(ramp_data):
                pos_turn_on = None
                for ind in range(len(ramp_data)):
                    if ramp_data[ind] > .0000001:
                        pos_turn_on = amp * (ind / (.25*num_pulses))
                        ind = num_pulses // 2
                    if pos_turn_on is not None and ramp_data[ind] < -.0000001:
                        neg_turn_on = -amp * (ind - .5*num_pulses) / (.25*num_pulses)
                        return pos_turn_on, neg_turn_on
                raise ValueError("No turning voltage found.")

        pos_turn_on_voltages = []
        if full_sweep:
            neg_turn_on_voltages = []

        for i in range(len(ramps)):
            pos_voltage, neg_voltage = find_turning_voltage(ramps[i])
            pos_turn_on_voltages.append(pos_voltage)
            if full_sweep:
                neg_turn_on_voltages.append(neg_voltage)

        plt.figure()
        plt.xlabel("Pulse Number")
        plt.ylabel("Turn On Voltage (V)")
        plt.plot(pos_turn_on_voltages)

        if full_sweep:
            plt.figure()
            plt.xlabel("Pulse Number")
            plt.ylabel("Turn On Voltage (V)")
            plt.plot(neg_turn_on_voltages)

        plt.legend()

        return ramps

    def process_access_device_square(self, file, header_lines, skip_footer=0, dwell_time_samples = 800, scale = 1e9,
                                     scale_text = "n", fig_prefix ='', save=False):
        """
        Processes the raw I-V curve of access device.
        Finds increasing and decreasing ramps, ignoring first half increasing ramp
        :param file: file to read IV data from
        :param header_lines: number of lines to ignore at top of file
        :param skip_footer: number of lines to ignore at end of file
        :param num_cycles: number of full cycles in voltage
        :param num_pulses: number of pulses per cycle
        :param amp: amplitude of the sweep, for a positive sweep this is maximum voltage
        :param full_sweep: true if sweep includes both positive and negative voltages, false if only positive

        :return:
        """

        data = np.genfromtxt(os.path.join(self.datapath, file), delimiter=',', skip_header=header_lines,
                             skip_footer=skip_footer)

        ramps = data[1::2]
        def get_average_current(ramp_data):
            dwell_time_currents = ramp_data[0:dwell_time_samples]
            return (np.sum(dwell_time_currents) / dwell_time_samples)

        average_currents = np.array([])

        for i in range(len(ramps)):
            average_currents = np.append(average_currents, (get_average_current(ramps[i]) * .1) - 3e-10)

        plt.figure()
        plt.xlabel("Pulse Number")
        plt.ylabel("Q (" + scale_text + "C)")
        plt.ylim(0, 12)
        plt.plot(average_currents*scale)
        plt.title("Diminishing Charge through Access Device")

        if save:
            plt.savefig(os.path.join(self.outdir, ".5v"), dpi=1200)

        return ramps

    def process_access_device_IV(self, file, header_lines, skip_footer=0, num_cycles=1, num_pulses=10000, fig_prefix="",
                                 save=False):
        """
        Processes the raw I-V curve of access device.
        Finds increasing and decreasing ramps, ignoring first half increasing ramp
        :param file: file to read IV data from
        :param header_lines: number of lines to ignore at top of file
        :param skip_footer: number of lines to ignore at end of file
        :param num_cycles: number of full cycles in voltage
        :param num_pulses: number of pulses per cycle
        :param fig_prefix: prefix to prepend saved figures with
        :param save: True if data should be saved

        :return:
        """

        data = np.genfromtxt(os.path.join(self.datapath, file), delimiter=',', skip_header=header_lines,
                             skip_footer=skip_footer)
        raw_fig, axarr = plt.subplots(2, sharex=True)
        plt.xlabel("Pulse Number")
        axarr[0].set_ylabel("Voltage")
        axarr[1].set_ylabel("Current")
        axarr[0].plot(data[:, 0])
        axarr[1].plot(data[:, 1])
        raw_fig.subplots_adjust(left=0.18, bottom=0.17)

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Raw_data"), dpi=1200)

        ramp_pulses = num_pulses // 2
        num_ramps = 2 * num_cycles - 1

        # Separate into increasing/decreasing ramps

        increasing_ramps = []
        decreasing_ramps = []

        # Ignore first half ramp up
        sample = data[ramp_pulses//2:]
        for i in range(num_ramps):
            # Check if ramp is increasing or decreasing
            if sample[0][0] > sample[ramp_pulses-1][0]:
                decreasing_ramps.append(sample[0:ramp_pulses])
            else:
                increasing_ramps.append(sample[0:ramp_pulses])
            sample = sample[ramp_pulses:]

        decreasing_ramps = decreasing_ramps[1:]

        # Plot ramps
        ramp_fig, axarr = plt.subplots(2, sharex=True, sharey=True)

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        axarr[0].set_title("Increasing ramps")

        # Plot with normal scale
        for i in range(len(increasing_ramps)):
            axarr[0].plot(np.transpose(increasing_ramps[i])[0], np.transpose(increasing_ramps[i])[1], label="Ramp %d" % i)
        box = axarr[0].get_position()
        axarr[0].set_position([box.x0, box.y0, box.width*0.8, box.height])
        axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend()

        axarr[1].set_title("Decreasing ramps")
        for i in range(len(decreasing_ramps)):
            axarr[1].plot(np.transpose(decreasing_ramps[i])[0], np.transpose(decreasing_ramps[i])[1], label="Ramp %d" % i)
        box = axarr[1].get_position()
        axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Ramps"), dpi=1200)

        # Plot with log scale
        ramp_fig, axarr = plt.subplots(2, sharex=True, sharey=True)

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.ylim([1e-7, 1e-3])
        axarr[0].set_title("Increasing ramps")

        for i in range(len(increasing_ramps)):
            axarr[0].semilogy(np.transpose(increasing_ramps[i])[0], np.abs(np.transpose(increasing_ramps[i])[1]), label="Ramp %d" % i)
        box = axarr[0].get_position()
        axarr[0].set_position([box.x0, box.y0, box.width*0.8, box.height])
        axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        axarr[1].set_title("Decreasing ramps")
        for i in range(len(decreasing_ramps)):
            axarr[1].semilogy(np.transpose(decreasing_ramps[i])[0], np.abs(np.transpose(decreasing_ramps[i])[1]), label="Ramp %d" % i)
        box = axarr[1].get_position()
        axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Ramps_Logarithmic"), dpi=1200)

        # Function for finding turn on/off voltage
        def find_turning_voltage(iv_data, increasing):
            on = None
            off = None
            for ind in range(len(iv_data)):
                i = iv_data[ind][1]
                if increasing:
                    if off is None and i > -.000001:
                        off = iv_data[ind][0]
                        off_idx = ind
                    if i > .000001:
                        on = iv_data[ind][0]
                        on_idx = ind
                        return on, off, on_idx, off_idx
                else:
                    if off is None and i < .000001 and ind > 100:
                        off = iv_data[ind][0]
                        off_idx = ind
                    if i < -.000001 and ind > 100:
                        on = iv_data[ind][0]
                        on_idx = ind
                        return on, off, on_idx, off_idx
            raise ValueError("No turning voltage found.")

        # Find all turn on/off voltages
        turn_on_incr = []
        turn_off_incr = []
        turn_on_decr = []
        turn_off_decr = []
        turn_on_idx_incr = []
        turn_off_idx_incr = []
        turn_on_idx_decr = []
        turn_off_idx_decr = []

        for i in range(len(increasing_ramps)):
            von, voff, i1, i2 = find_turning_voltage(increasing_ramps[i], increasing=True)
            turn_on_incr.append(von)
            turn_on_idx_incr.append(i1)
            turn_off_incr.append(voff)
            turn_off_idx_incr.append(i2)

        for i in range(len(decreasing_ramps)):
            von, voff, i1, i2 = find_turning_voltage(decreasing_ramps[i], increasing=False)
            turn_on_decr.append(von)
            turn_on_idx_decr.append(i1)
            turn_off_decr.append(voff)
            turn_off_idx_decr.append(i2)

        def plot_pdf(dat, label=""):
            kde = gaussian_kde(dat, 1)
            x = np.linspace(np.min(dat), np.max(dat), 100)
            plt.plot(x, kde(x), 'r-', lw=2, alpha=.6)
            plt.hist(dat, normed=True, alpha=.2, label=label)
            # sorted_data = np.sort(data)
            # cdf = np.linspace(0, 1, len(sorted_data))
            # plt.plot(sorted_data, cdf, label=label)

        plt.figure()
        plt.title("Positive On/Off Voltages CDF")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Probability")
        plot_pdf(turn_on_incr, label="Turn on voltage")
        plot_pdf(turn_off_decr, label="Turn off voltage")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Positive_On_Off_Voltage_CDF"), dpi=1200)

        plt.figure()
        plt.title("Negative On/Off Voltages CDF")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Probability")
        plot_pdf(turn_on_decr, label="Turn on voltage")
        plot_pdf(turn_off_incr, label="Turn off voltage")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Negative_On_Off_Voltage_CDF"), dpi=1200)

        # Calculate resistance when on

        resistances = np.array([])

        def find_resistances(ramp, ons, offs):
            res = np.array([])
            for i in range(len(ramp)):
                # Find resistance when still on from previous ramp
                ys = np.transpose(ramp[i])[1][0:offs[i]]
                xs = np.transpose(ramp[i])[0][0:offs[i]]
                slope1, _, _, _, _ = stats.linregress(xs, ys)
                res = np.append(res, slope1)

                # Find resistance after turning on
                ys = np.transpose(ramp[i])[1][ons[i]:-1]
                xs = np.transpose(ramp[i])[0][ons[i]:-1]
                slope2, _, _, _, _ = stats.linregress(xs, ys)
                res = np.append(res, slope2)
            return res

        resistances = np.append(resistances, find_resistances(increasing_ramps, turn_on_idx_incr, turn_off_idx_incr))
        resistances = np.append(resistances, find_resistances(decreasing_ramps, turn_on_idx_decr, turn_off_idx_decr))

        # Plot resistance cdf
        plt.figure()

        plt.xlabel("Resistance (mOhms)")
        plt.ylabel("Probability")
        plt.title("Resistance CDF")
        plot_pdf(resistances*1000, label="Resistance (mOhms)")
        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"Resistance_CDF"), dpi=1200)

        return turn_on_incr, turn_off_incr

    def find_ENODe_ramps(self, file, Gmin, Gmax):
        data = np.genfromtxt(os.path.join(self.datapath, file), delimiter=',', skip_header=0,
                             skip_footer=0)
        conductances = np.transpose(data)[1]
        increasing_ramps = []
        decreasing_ramps = []

        min_conductance_idxs = []
        max_conductance_idxs = []
        increasing = False
        for i in range(len(conductances)):
            if conductances[i] > Gmax and increasing:
                max_conductance_idxs.append(i)
                increasing = False
            if conductances[i] < Gmin and not increasing:
                min_conductance_idxs.append(i)
                increasing = True

        # Assuming starts at decreasing ramp
        # Decreasing ramps go from a min idx to a max idx, number of ramps is equal to min of number of min/max idxs
        num_decreasing = min(len(max_conductance_idxs), len(min_conductance_idxs))
        for i in range(num_decreasing):
            decreasing_ramps.append(conductances[min_conductance_idxs[i]:max_conductance_idxs[i]])
        # Increasing ramps go from a max idx to a min idx, number of ramps is 1 less than number of min idxs
        for i in range(len(min_conductance_idxs) - 1):
            increasing_ramps.append(conductances[max_conductance_idxs[i]:min_conductance_idxs[i+1]])
        return increasing_ramps, decreasing_ramps


    def find_ENODe_ramps_separate(self, incr_file, decr_file, Gmin, Gmax):
        incr_data = np.genfromtxt(os.path.join(self.datapath, incr_file), delimiter=',', skip_header=0,
                             skip_footer=0)
        incr_conductances = np.transpose(incr_data)[1]
        decr_data = np.genfromtxt(os.path.join(self.datapath, decr_file), delimiter=',', skip_header=0,
                             skip_footer=0)
        decr_conductances = np.transpose(decr_data)[1]
        increasing_ramps = []
        decreasing_ramps = []

        min_conductance_idxs = []
        max_conductance_idxs = []
        i = 0
        while len(incr_conductances):
            if incr_conductances[i] < Gmin:
                increasing_ramps.append(incr_conductances[0:i+1])
                incr_conductances = incr_conductances[i+1:]
                i = 0
            else:
                i += 1

        while len(decr_conductances):
            if decr_conductances[i] > Gmax:
                decreasing_ramps.append(decr_conductances[0:i+1])
                decr_conductances = decr_conductances[i+1:]
                i = 0
            else:
                i += 1

        return increasing_ramps, decreasing_ramps



    def plot_GvsdG(self, incr_x, incr_y, incr_z, decr_x, decr_y, decr_z, scale = 1e6):
        fig = plt.figure(figsize=(5, 2.5))

        plt.subplot(1, 2, 1)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel(r"|$\Delta$G| (" + "$\mu$" + "S)", labelpad=1, fontsize=16)
        surf = plt.contourf(incr_x * scale, incr_y * scale, incr_z.T, levels=np.linspace(0, 1, 501), cmap='RdYlBu')
        plt.xlim([incr_x[0] * scale, incr_x[-1] * scale])
        plt.title("Depression", fontsize=10)

        plt.subplot(1, 2, 2)
        plt.xticks(fontsize=10)
        ax = plt.gca()
        ax.set_yticklabels([])
        plt.contourf(decr_x * scale, decr_y * scale, decr_z.T, levels=np.linspace(0, 1, 501), cmap='RdYlBu')
        plt.xlim([decr_x[0] * scale, decr_x[-1] * scale])
        plt.title("Potentiation", fontsize=10)
        fig.text(.5, .05, "Conductance (" + "$\mu$" + "S)", ha='center', va='center', fontsize=16)

        fig.subplots_adjust(bottom=0.2, right=.8)
        colorbar_axes = fig.add_axes([.85, .2, .05, .675])

        cbar = fig.colorbar(surf, ticks=np.linspace(0, 1, 6), cax=colorbar_axes)
        cbar.set_label("Probability Density")
        plt.savefig(os.path.join(self.outdir, "G_vs_dG.png"), dpi=1200)



# ***************** Data Processing Functions***************************************

    def bin_data(self,pulse_data, Gmin, Gmax, nbins = 50,fig_prefix  = "", extrapolate = False, abs=False, save=True, dGmin=None, dGmax = None):
        """
        Returns binned data:  bins is the G value of each bin
        binned_dG is the sorted dG values in each bin (list of numpy arrays)
        binned_CDF is the CDF at each dG value in binned_dG (list of numpy arrays)

        :param Gmin: the minimum conductance
        :param Gmax: the maximum conductance
        :param pulse_data:
        :type pulse_data: list
        :param fig_prefix: A prefix to append to figure filenames before saving
        :param extrapolate: if False, dG is set to the last measured value of G values outside the measured range
                            if true, extrapolate the dG outside the measured range
        :param dGmin:  Minimum dG for plotting, if none, derive from data
        ":param dGmax:  Maximum dG for plotting, if none, derive from data
        :return:binned_dG, binned_CDF, bins
        """

        dG = []

        #create bins for G
        G_bins, bin_spacing = np.linspace(Gmin,Gmax,nbins+1, endpoint=True, retstep=True)
        binned_dG = [ [] for _ in range(len(G_bins))] # for each conductance bin, stores dG values
        binned_CDF = [ [] for _ in range(len(G_bins))]# for each conductance bin, stores CDF values corresponding to dG value above



        for ind in range(len(pulse_data)):
            # compute delta G and interpolate delta G into all bins
            dG = np.diff(pulse_data[ind])

            # find the midpoints (assign each dG to the midpoint between the start and end)
            midpts = (pulse_data[ind][0:-1]+pulse_data[ind][1:])/2

            

            # interpolate dG to all bin values
            if extrapolate:
                dG_interpolator2= sp.interpolate.interp1d(midpts,dG,kind='linear',bounds_error=False, fill_value='extrapolate')
                # limit interpolation to a maximum of twice the min/max dG values
                # only interpolate at the start of the ramp, assume pulses saturate at the end of a ramp
                if pulse_data[ind][0]<pulse_data[ind][-1]:  # get max/min values right depending on whether we have increasing or decreasing pulses
                    ramp_direction = "increasing"
                    ramp_limit = midpts.max()*0.9999
                else:
                    ramp_direction = "decreasing"
                    ramp_limit = midpts.min()*1.0001

                def dG_interpolator(data):
                    data = data.copy()
                    if ramp_direction=="increasing":
                        data[data>ramp_limit]=ramp_limit
                    else:
                        data[data<ramp_limit]=ramp_limit

                    result = dG_interpolator2(data)

                    dG_max_local = 2*np.amax(dG)
                    dG_min_local = 2*np.amin(dG)
                    result[result>dG_max_local]=dG_max_local
                    result[result<dG_min_local]=dG_min_local
                    return result
            else:
                if pulse_data[ind][0]<pulse_data[ind][-1]:  # get max/min values right depending on whether we have increasing or decreasing pulses
                    fill_value=(dG[0],dG[-1])
                else:
                    fill_value=(dG[-1],dG[0])

                dG_interpolator= sp.interpolate.interp1d(midpts,dG,kind='linear',bounds_error=False, fill_value=fill_value)

            dG_values = dG_interpolator(G_bins)

            # G_min = min([pulse_data[ind][0],pulse_data[ind][-1]])
            # G_max = max(pulse_data[ind][0],pulse_data[ind][-1])
            G_min = min(pulse_data[ind])
            G_max = max(pulse_data[ind])

            #for each pulse train, for each conductance bin, add dG
            for ind2 in range(len(binned_dG)-1):
                if (G_bins[ind2]>=G_min) and (G_bins[ind2]<=G_max):  #only add bins if they are between Gmin and Gmax
                    binned_dG[ind2].append(dG_values[ind2])





        #find dG limits
        dGmin1, dGmax1 = self.find_min_max(binned_dG)

        if dGmin is None: dGmin = dGmin1
        if dGmax is None: dGmax = dGmax1


        # *******  create dG vs G plot
        #create X, Y, Z vectors to plot
        X_G=np.array([G_bins[0],G_bins[0],G_bins[-1],G_bins[-1]])
        if abs:
            binned_dG = [np.abs(row) for row in binned_dG]
            dGmin = 0e-6
            dGmax = 5e-6

        Y_dG = np.array([dGmin,dGmax,dGmin,dGmax])
        Z_CDF=np.array([0,1,0,1])

        # sort binned data
        fig = plt.figure()
        for ind in range(len(binned_dG)):
            binned_dG[ind] = np.sort(binned_dG[ind])
            num_dG = len(binned_dG[ind])
            binned_CDF[ind] = (np.array(range(num_dG))+0.5)/num_dG
            plt.plot(binned_dG[ind],binned_CDF[ind])

            # create plotting lists
            X_G=np.append(X_G, (G_bins[ind])*np.ones(len(binned_dG[ind]))) #adjust x to be the center of the bin
            Y_dG=np.append(Y_dG,binned_dG[ind])
            Z_CDF=np.append(Z_CDF,binned_CDF[ind])


        plt.xlabel("$\Delta$G (S)", labelpad=1)
        plt.ylabel("CDF", labelpad=1)
        plt.title("CDF vs dG for different G bins")


        # create G vs dG plot


        ####### interpolate onto a regular grid (so interpolation can be controlled and pdf can be created)

        # create vectors to interpolate to
        X_G_list =G_bins[1:-1]  #remove 1st and last bins
        Y_dG_list = np.linspace(dGmin,dGmax,40)

        #scaling to set relative scales for interpolation
        X_G_scaling = X_G_list[-1]/10
        Y_dG_scaling = np.max(np.abs(Y_dG_list))

        points = np.array((X_G/X_G_scaling, Y_dG/Y_dG_scaling)).T
        interpolator = LinearNDInterpolator(points, Z_CDF, fill_value=np.nan)


        Z_CDF_array = np.zeros([len(X_G_list),len(Y_dG_list)])


        # do interpolation
        for ind in range(len(X_G_list)):
            X = X_G_list[ind]*np.ones(len(Y_dG_list))/X_G_scaling # normalize to max value to help interpolation
            Y = Y_dG_list/Y_dG_scaling
            points = np.array((X, Y)).T
            Z_list = interpolator(points)

            # replace NaNs with closest defined value in each G column
            mask = np.isnan(Z_list)
            if np.all(mask):
                Z_CDF_array[:,ind]=0.5  # if all values are NaN, use 0.5
            elif np.any(mask):
                Z_list[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Z_list[~mask])
                Z_CDF_array[ind,:]=Z_list
            else:
                Z_CDF_array[ind,:]=Z_list



        #########  plot scatterplot
        fig = plt.figure()


        dG_cumulative = np.array([])
        G_start_cumulative = np.array([])
        for ind in range(len(pulse_data)):
            # compute delta G and place delta G in all bins spanned by delta G  (allows for really large steps)
            dG_cumulative=np.append(dG_cumulative ,np.diff(pulse_data[ind]) )
            G_start_cumulative=np.append(G_start_cumulative,pulse_data[ind][:-1])
        plt.plot(G_start_cumulative*self.scale, dG_cumulative*self.scale,'o',markersize=2, markeredgewidth=0, alpha=0.25)

        if self.G_ticks is not None:
            plt.xticks(self.G_ticks)
            plt.xlim([Gmin*self.scale, Gmax*self.scale])
        else:
            plt.locator_params(axis='x',nbins=5)

        plt.xlabel(r"Initial Conductance ("+self.scale_text+"S)", labelpad=0.5)
        plt.ylabel(r"$\Delta$G ("+self.scale_text+"S)", labelpad=0.5)
        # plt.title(fig_prefix+" G vs dG")
        plt.ylim([dGmin*self.scale,dGmax*self.scale])

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"_G_vs_dG_scatter.png"),dpi=1200)




        ######## create CDF G vs dG plot
        fig = plt.figure()


        surf=plt.contourf(X_G_list*self.scale,Y_dG_list*self.scale,Z_CDF_array.T, levels=np.linspace(0,1.0,501),cmap='RdYlBu')

        cbar = fig.colorbar(surf, ticks = np.linspace(0,1,6))
        cbar.set_label("CDF")

        plt.xlim([G_bins[1]*self.scale,G_bins[-2]*self.scale])
        plt.ylim([dGmin*self.scale,dGmax*self.scale])
        if self.G_ticks is not None:
            plt.xticks(self.G_ticks)
            plt.xlim([X_G_list[0]*self.scale, X_G_list[-1]*self.scale])
        else:
            plt.locator_params(axis='x',nbins=5)


        plt.xlabel(r"Conductance ("+self.scale_text+"S)", labelpad=1)
        plt.ylabel(r"$\Delta$G ("+self.scale_text+"S)", labelpad=1)
        # plt.title(fig_prefix + " G vs dG CDF")

        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"_G_vs_dG.png"),dpi=1200)
        # plt.savefig(os.path.join(self.outdir, fig_prefix+"_G_vs_dG.eps"),format='eps',dpi=1200)


        ###### create PDF plot
        Z_PDF_array = np.diff(Z_CDF_array,axis=1)
        Z_PDF_array[np.isnan(Z_PDF_array)]=0
        Z_PDF_array[Z_PDF_array<0]=0

        fig = plt.figure()

        surf=plt.contourf(X_G_list*self.scale,Y_dG_list[0:-1]*self.scale,Z_PDF_array.T, levels=np.linspace(0.01,0.5,501),cmap='BuGn')

        cbar = fig.colorbar(surf, ticks = np.linspace(0,0.5,6))
        cbar.set_label("Probability Density")

        plt.xlim([G_bins[1]*self.scale,G_bins[-2]*self.scale])
        plt.ylim([dGmin*self.scale,dGmax*self.scale])
        plt.xlabel(r"Conductance ("+self.scale_text+"S)", labelpad=1)
        plt.ylabel(r"$\Delta$G ("+self.scale_text+"S)", labelpad=1)
        # plt.title(fig_prefix + " G vs dG PDF")
        if self.G_ticks is not None:
            plt.xticks(self.G_ticks)
            plt.xlim([X_G_list[0]*self.scale, X_G_list[-1]*self.scale])
        else:
            plt.locator_params(axis='x',nbins=5)


        if save:
            plt.savefig(os.path.join(self.outdir, fig_prefix+"_G_vs_dG_PDF.png"),dpi=500)


        return binned_dG, binned_CDF, G_bins#, X_G_list, Y_dG_list, Z_CDF_array


    def create_lookup_table(self,binned_dG, binned_CDF, Gbins,n_CDF_points=501, n_CDF_end_points=0, filename="dG_lookuptable.txt", n_G_points = 501, filter_CDF = 0., fig_prefix  = ""):
        """

        :param binned_dG: the delta G valeus in each bin
        :param binned_CDF: the binned CDF values
        :param Gbins:  the G bins for the data
        :param n_CDF_points: number of linearly distributed cdf points
        :param n_CDF_end_points:  # extra points at the end that are logarithmically distributed
        :param filename:  the output filename to save to
        :param n_G_points: number of G points in the output.  If None, the default bin spacing is used, otherwise interpolated to match requested # points
        :param filter_CDF: removes events with Probability<filter_CDF. Used to filter out rare events / experiemental error
        :return:
        """


        # create vector of CDF points to interpolate to (interpolate points now for easy interpolation in the lookup table)
        CDF_points, CDF_spacing = np.linspace(0,1,n_CDF_points,retstep=True)
        if n_CDF_end_points !=0:  # insert extra end points if desired
            start_points = np.exp2(np.linspace(-n_CDF_end_points,-1,n_CDF_end_points+1))*CDF_spacing
            end_points = 1-start_points
            end_points = end_points[::-1]
            CDF_points = np.insert(CDF_points,1,start_points)
            CDF_points = np.insert(CDF_points,-1,end_points)

        # trim the initial bins from binned data (are empty / out of range bins), also trim first and last bin to eliminate edge effects
        binned_dG=binned_dG[2:-1]
        binned_CDF=binned_CDF[2:-1]
        Gbins_data=Gbins[2:-1]

        if n_G_points is None:
            Gbins =Gbins_data
        else:
            Gbins = np.linspace(Gbins_data[0],Gbins_data[-1],n_G_points)


        #create numpy array to hold interpolated dG
        dG_array = np.zeros([len(CDF_points),len(Gbins)])

        if n_G_points is None:
            for ind in range(len(Gbins)):
                #set dG to zero for empty bins
                if len(binned_CDF[ind])==0:
                    dG_array[:,ind]=0
                else:
                    dG_array[:,ind]=np.interp(CDF_points,binned_CDF[ind],binned_dG[ind])
        else:
            # create X, Y, Z array for interpolation
            X = np.array([])
            Y = np.array([])
            Z = np.array([])
            for ind in range(len(Gbins_data)):
                G = Gbins_data[ind]/Gbins_data[-1]*2 # normalize to max value to help interpolation
                X = np.append(X, G*np.ones(len(binned_CDF[ind])))
                Y = np.append(Y, binned_CDF[ind])
                Z = np.append(Z, binned_dG[ind])

            # find average dG value to fill in for missing values
            mean_dG = np.mean(Z)

            points = np.array((X, Y)).T

            interpolator = LinearNDInterpolator(points, Z, fill_value=np.nan)

            # do interpolation
            for ind in range(len(Gbins)):
                X = Gbins[ind]*np.ones(len(CDF_points))/Gbins_data[-1]*2 # normalize to max value to help interpolation
                Y = CDF_points
                points = np.array((X, Y)).T
                dG_list = interpolator(points)

                # filter CDF endpoints if requested by replacing with nans (which are then replaced with the colsest value below)
                if filter_CDF !=0:
                    ind_min = (np.abs(CDF_points-filter_CDF)).argmin()
                    ind_max = (np.abs(1-filter_CDF-CDF_points)).argmin()
                    dG_list[0:ind_min+1] = np.nan
                    dG_list[ind_max:] = np.nan


                # replace NaNs with closest defined value in each G column
                mask = np.isnan(dG_list)
                if np.all(mask):
                    dG_array[:,ind]=mean_dG  # if all values are NaN, use mean dG value
                elif np.any(mask):
                    dG_list[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), dG_list[~mask])
                    dG_array[:,ind]=dG_list
                else:
                    dG_array[:,ind]=dG_list



        ##### create plot of G vs CDF vs dG

        fig = plt.figure()

        surf = plt.pcolormesh(Gbins*self.scale,CDF_points,dG_array*self.scale,shading='auto')
        cbar = fig.colorbar(surf)
        plt.xlabel("Conductance ("+self.scale_text+"S)",labelpad=1)
        plt.ylabel("Cumulative Probability",labelpad=1)
        cbar.set_label("$\Delta$G ("+self.scale_text+"S)")
        if self.G_ticks is not None:
            plt.xticks(self.G_ticks)
        else:
            plt.locator_params(axis='x',nbins=5)
        plt.xlim([Gbins[0]*self.scale,Gbins[-1]*self.scale])


        fig.subplots_adjust(left=0.15, bottom=0.17, right=0.85)

        plt.savefig(os.path.join(self.outdir, fig_prefix+"_G_vs_CDF.png"),dpi=500)

        #####  save results to file
        #create header strings with Gbins and CDF points
        header = "1st Row = Gbins, 2nd Row = CDF Points, 3rd row onward =  dG matrix (rows= CDF, columns = Gbins) \n"
        header1 = np.array2string(Gbins,separator=',',max_line_width=1e20, precision=16)
        header+=header1[1:-1]
        header+='\n'
        header2 = np.array2string(CDF_points,separator=',',max_line_width=1e20, precision=16)
        header+=header2[1:-1]
        header+='\n'

        np.savetxt(os.path.join(self.outdir, filename), dG_array,delimiter=',',header=header,comments='')

        return Gbins,CDF_points,dG_array
