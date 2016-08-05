#!/usr/bin/env ipython
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.style.use('ggplot')
#matplotlib.style.use('bmh')
matplotlib.rcParams.update({'font.size': 12})

def print_progress(count, n_events):
    percent = float(count) / n_events
    hashes = '#' * int(round(percent * 20))
    spaces = ' ' * (20 - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

# Take filename from terminal input. Assumes <file> pairs with <file>_ped
data_name     = sys.argv[1] + '_' + sys.argv[2]
data_ped_name = data_name + '_ped'
# Match with pickle_data        #TODO: make this automatic
n_samples = 402
frameSize = 10.05e-8  # Not actually used in the analysis currently, but can be useful to know

data           = pd.read_pickle('data/' + data_name+'.pkl')
data_ped       = pd.read_pickle('data/' + data_ped_name+'.pkl')

charge_subset = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

n_events = len(data)/n_samples

print "time interval:", dt
print "number of events:", n_events

# This part would only be used if we wanted to define some range of integration (by default we just take everything)
low_offset = 0
high_offset = 0
# Let's make some assumptions about the rise-time
rise_time = 1e-9 #seconds
rise_time_in_samples = int(rise_time/dt)
fall_time = 4e-9 #seconds
fall_time_in_samples = int(fall_time/dt)

# Define range of integration
filterTimeRange = True
lower_time = 40.
upper_time = 70.
if filterTimeRange:
    data = data[(np.abs(data.filtered_voltage)<90) & (data.time > lower_time) & (data.time < upper_time)]
    data_ped = data_ped[(np.abs(data_ped.filtered_voltage)<90) & (data_ped.time > lower_time) & (data_ped.time < upper_time)]
else:
    data = data[(np.abs(data.filtered_voltage)<50)]
    data_ped = data_ped[(np.abs(data_ped.filtered_voltage)<50)]

# Shift baseline to zero
mean_ped_voltage = data_ped.filtered_voltage.mean()
print "Offet all voltages by the average baseline voltage of the pedestal:", mean_ped_voltage
data_ped.voltage          = data_ped.voltage - mean_ped_voltage
data_ped.filtered_voltage = data_ped.filtered_voltage - mean_ped_voltage
data.voltage              = data    .voltage - mean_ped_voltage
data.filtered_voltage     = data    .filtered_voltage - mean_ped_voltage

# Apply Time Over Threshold Filter
voltage_threshold = -3.
min_TOT = data[(data.filtered_voltage < voltage_threshold)].groupby(['eventID']).time.min()
max_TOT = data[(data.filtered_voltage < voltage_threshold)].groupby(['eventID']).time.max()
diff = (max_TOT - min_TOT) > 0.7 # 700ps
diff = diff[diff] # only select the events where the above condition is true
good_data = data[data.eventID.isin(diff.index)]
print "Number of good pulses above threshold (-1 mV) is:", len(diff), len(good_data), len(data)
bad_data = data[-(data.eventID.isin(diff.index))]
print "Bad data", bad_data.shape[0]

# Restrict the good data to within the time above the threshold. Only integrate in this window
good_data = good_data[(good_data.time > min_TOT[good_data.eventID]) & (good_data.time < max_TOT[good_data.eventID])]

# Group the data into events (i.e., separate triggers)
grouped_data      = data     .groupby(['eventID'])
grouped_data_ped  = data_ped .groupby(['eventID'])
grouped_data_good = good_data.groupby(['eventID'])
grouped_data_bad  = bad_data .groupby(['eventID'])

# Plot the time position and voltage of the max voltage in each event
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
max_voltages     = data    .loc[grouped_data    .filtered_voltage.idxmin()]
max_voltages_ped = data_ped.loc[grouped_data_ped.filtered_voltage.idxmin()]
max_voltages    .plot(kind='scatter',x='time',y='filtered_voltage', title = 'LED on',  ax = axes[0])
max_voltages_ped.plot(kind='scatter',x='time',y='filtered_voltage', title = 'LED off', ax = axes[1])
axes[0].set_xlabel("time [ns]")
axes[1].set_xlabel("time [ns]")
axes[0].set_ylabel("filtered voltage [mV]")
axes[0].set_ylabel("")
fig.savefig('plots/' + sys.argv[1] + sys.argv[2] + 'max_voltage_vs_time.png')

# Plot the amplitude of each signal in a histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
axes[0].set_yscale('log')
axes[1].set_yscale('log')
max_voltages['minus_voltage'] = -1*max_voltages['voltage']
max_voltages_ped['minus_voltage'] = -1*max_voltages_ped['voltage']

max_voltages['minus_voltage']    .hist(histtype='step', bins = 50, color='r', ax = axes[0])
max_voltages_ped['minus_voltage'].hist(histtype='step', bins = 50, color='r', ax = axes[1])
fig.savefig('plots/' + sys.argv[1] + sys.argv[2] + 'signal_amplitude_spectrum.png')

# Convert the voltage into a collected charge and sum over all voltages in the time window
scale  = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
q      = scale*grouped_data     .filtered_voltage.sum()
q_ped  = scale*grouped_data_ped .filtered_voltage.sum()
q_good = scale*grouped_data_good.filtered_voltage.sum()
q_bad  = scale*grouped_data_bad .filtered_voltage.sum()
print "Q bad", q_bad.shape[0]

# Fit a normal distribution to the pedestal
from scipy.stats import norm
mu, std = norm.fit(q_ped.values)
mu_ped  = q_ped.mean()
std_ped = q_ped.std()
print
print "Fitted mean and standard deviation of the pedestal    : %0.3f, %0.3f  " % (mu, std)
print "Calculated mean and standard deviation of the pedestal: %0.3f, %0.3f\n" % (mu_ped, std_ped)

# Plot the spectrum of collected charge
loC = -1
hiC =  4.
nBins = 200
width = float(hiC-loC)/nBins
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].set_yscale('log')
axes[1].set_yscale('log')
q     .hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='r', ax = axes[0])
q_good.hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='g', ax = axes[0])
q_ped .hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='b', ax = axes[1])

# uncomment these lines if you want to plot the fitted Gaussian
#x = np.linspace(loC, hiC, nBins)
#pdf = n_events*norm.pdf(x, mu, std)/nBins
#plt.plot(x, pdf, 'k', linewidth = 2)

axes[0].set_xlabel("charge [pC]")
axes[0].set_ylabel("Entries / (%0.2f pC)" % width)
axes[0].set_xlim(loC, hiC)
axes[0].set_ylim(1, 1e4)
axes[1].set_xlabel("charge [pC]")
axes[1].set_ylabel("Entries / (%0.2f pC)" % width)
axes[1].set_xlim(loC, hiC)
axes[1].set_ylim(1, 1e4)
fig.savefig('plots/' + sys.argv[1] + sys.argv[2] + 'charge_spectrum.png')

# Compute the mean of the collected charge above some threshold (which is defined in terms of the pedestal) to compute the gain
from math import sqrt
threshold = 3*std_ped
signal = q[q > threshold]
mean_signal_charge = signal.mean()
gain0     = mean_signal_charge*1.e-12/1.602e-19/1e7
gain0_err = mean_signal_charge*1.e-12/1.602e-19/1e7/sqrt(len(signal)) # i.e. error on the mean
gain      = q_good.mean()*1.e-12/1.602e-19/1e7
gain_err  = gain/sqrt(len(q_good))

print "Using time-over-threshold:             gain of the photo-sensor is (%0.3f +- %0.3f) x 10^7\n" % (gain, gain_err)
print "Using 3-sigma threshold from pedestal: gain of the photo-sensor is (%0.3f +- %0.3f) x 10^7\n" % (gain0, gain0_err)

# Now show the oscillioscope traces of a few events
subset1 = (q[(q > 0.2) & (q < 0.5)]).sample(n=4).index
subset2 = (q[(q > 0.5) & (q < 1.0)]).sample(n=4).index
subset3 = (q[(q > 0.5) & (q < hiC)]).sample(n=4).index


if len(subset3) > 0:
    trace, trace_ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 10))
    grouped_data.get_group(subset3[0]).plot(x='time',y='voltage'         ,ax=trace_ax, legend=False)
    grouped_data.get_group(subset3[0]).plot(x='time',y='filtered_voltage',ax=trace_ax, legend=False)
    trace_ax.set_xlabel("time [ns]")
    trace_ax.set_ylabel("voltage [mV]")
    trace.savefig('plots/' + sys.argv[1] + sys.argv[2] + 'oscilloscope_traces_single.png')

if len(subset1) > 0 and len(subset2) > 0 and len(subset3) > 0:
    trace, trace_ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20, 10))
    for i in range(4):
        grouped_data.get_group(subset1[i]).plot(x='time',y='voltage'         ,ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset1[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='voltage'         ,ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='voltage'         ,ax=trace_ax[2,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[2,i], legend=False)
        ''' TODO: gives error, not sure why - look in to this.
        trace_ax[0,i].axvline(min_TOT[subset1[i]])
        trace_ax[0,i].axvline(max_TOT[subset1[i]])
        trace_ax[1,i].axvline(min_TOT[subset2[i]])
        trace_ax[1,i].axvline(max_TOT[subset2[i]])
        trace_ax[2,i].axvline(min_TOT[subset3[i]])
        trace_ax[2,i].axvline(max_TOT[subset3[i]])
        '''
        '''
        # This plots the points defining the integration region
        data[bounds[interesting [k ]][0]:bounds[interesting [k ]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting [k ]][1]:bounds[interesting [k ]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting2[k2]][0]:bounds[interesting2[k2]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting2[k2]][1]:bounds[interesting2[k2]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting3[k3]][0]:bounds[interesting3[k3]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        data[bounds[interesting3[k3]][1]:bounds[interesting3[k3]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        '''
        trace_ax[0,i].set_xlabel("")
        trace_ax[1,i].set_xlabel("")
        trace_ax[2,i].set_xlabel("time [ns]")
        trace_ax[0,0].set_ylabel("voltage [mV]")
        trace_ax[1,0].set_ylabel("voltage [mV]")
        trace_ax[2,0].set_ylabel("voltage [mV]")
        trace.savefig('plots/' + sys.argv[1] + sys.argv[2] + 'oscilloscope_traces.png')