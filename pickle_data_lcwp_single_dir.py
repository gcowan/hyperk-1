#!/usr/bin/env ipython
import pandas as pd
import sys
import os
from scipy import signal

# Defaults come from TekTronix 'scope
def filter_signal(data, nSamples = 1000, frameSize = 40e-9):
    '''Butterworth filter of signal.'''
    fs = float(nSamples)/frameSize # sample rate (1000 samples in 40ns)
    nyq = fs*0.5 # Nyquist frequency
    high = 700e6/nyq # high frequency cut-off
    b, a = signal.butter(2, high, 'low', analog = False)
    y = signal.lfilter(b, a, data)
    return y

# Conversion factors      # Cn:INSPECT?
vert_gain   = float(sys.argv[3]) #6.1035e-07   # "VERTICAL_GAIN"
vert_offset = -5.700e-2   # "VERTICAL_OFFSET"
vert_units  = 'V'         # "VERTUNIT"
horiz_units = 's'         # "HORUNIT"
horiz_scale = 2.500e-10   # "HORIZ_INTERVAL"

# Data characteristics
n_samples = 402          # TODO: find this automatically
frameSize = n_samples*horiz_scale
print "Frame size:", frameSize

# Read folder names from directory
#measurements = [d for d in os.listdir('' + sys.argv[1] + '/') if (d[0]!='.' and (not (d.endswith('.sh') or d.endswith('.log') or d.endswith('.pkl') or d.endswith('.txt'))))] # Filters out system files
measurements = [sys.argv[1]]
print measurements
n_measurements = len(measurements)
measurement_count = 1


for measurement in measurements:
    print('Run %d of %d' % (measurement_count, n_measurements))

    # Initialise dictionaries for producing final DataFrame
    output_dict     = {}

    output_dict['time']                 = []
    output_dict['eventID']              = []
    output_dict['voltage']              = []
    output_dict['filtered_voltage']     = []

    # Produce time index
    index = [i * horiz_scale * 1e9 for i in range(n_samples)]  # Convert to ns

    # Read filenames from inner directory
    #files     = [x for x in os.listdir('' + sys.argv[1] + '/' + measurement)          if x.endswith(sys.argv[2]+'.txt')]
    files     = [x for x in os.listdir(measurement)          if x.endswith(sys.argv[2]+'.txt')]
    n_files = len(files)
    file_count = 0
    # Loop through each file_pair to extract events
    for file in files:
        #data     = open((''+sys.argv[1]+'/'+measurement+    '/'+file),'r').read()
        data     = open((measurement+    '/'+file),'r').read()

        n_points = int(len(data)/4)
        n_events = int(n_points/n_samples)
        # Convert HEX to mV
        dec_data = []
        for i in range(n_points):
            dec_value     = int(data[i*4:(i*4)+4], 16)

            dec_data    .append((dec_value*vert_gain + vert_offset)*1e3) # to mV

        # Separate into events
        for i in range(n_events):
            output_dict['time']    .extend(index)

            output_dict['eventID']    .extend([i + file_count*n_events] * n_samples)

            voltages     = []
            for j in range(n_samples):
                voltages    .append(dec_data[i*n_samples+j])
            output_dict['voltage']    .extend(voltages)

            # Apply Butterworth filter
            output_dict['filtered_voltage']    .extend(filter_signal(voltages,     n_samples, frameSize))

        file_count += 1

    measurement_count += 1

    # Convert dictionaries to df then pickle
    output_df     = pd.DataFrame.from_dict(output_dict)

    print "Events:", output_df.shape[0]/n_samples

    output_df    .to_pickle(measurement + '_' + sys.argv[2]+'.pkl')
