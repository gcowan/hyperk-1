# hyperk
Repository for analysis code edited during the LAPPD test summer project

pickle_data_lcwp.py:
WHAT DOES IT DO? 
Takes the output hexadecimal files from the LeCroy 'scope and converts them
to a pkl file, capable of being read by make_spectra.py in the same way as 
the TekTronix files.
HOW TO RUN IT?
python pickle_data_lcwp.py <directory> <channel>
Where directory is the sub-folder within a folder called 'data' in your current
working directory. This can contain any number of files, but must match with 
the related <directory>_ped file. Channael can be 'C1' or 'C2' and relates to
the file naming convention used in the LabView driver.
PARAMETERS TO CONSIDER?
Within the code, there are a number of parameters at the start to think about.
The four conversion factors at the top - VERT_OFFSET, VERT_GAIN, etc - are
returned in the "Instrument Parameters" box on LabView. Also important is 
n_samples: depending on the time base you use this could be 202, 402, 802 etc.
The horizontal acquisition setup screen will tell you how many points. This 
will also require alteration of a parameter in lab view - either 200, 400, 800
etc. 
WHAT DOES IT RETURN?
A file called data/<directory>_<channel>.pkl and another called 
data/<directory>_<channel>_ped.pkl, both of which are required for make_spectra
POTENTIAL IMPROVEMENTS?
An input file containing information on the 'scope parameters would save some
time and perhaps reduce the chance of error - this could be produced using 
LabView. 
Automation through an entire directory of different measurements - e.g. various
HV values when taking a gain curve - is simple to implement. See some of the 
TekTronix pickle codes for an example.
It should be possible to read n_samples automatically - either from the LabView
parameter file, or from the input file directly.

pickle_data_lcwp_dir.py
This does the same as above, but the <directory> argument concerns a folder
containing an entire set of measurments. This will loop through this directory
and output a pickle signal and pedestal file for each sub-folder within - i.e.
each measurement taken in the run.

make_spectra.py
WHAT DOES IT DO?
Takes in a pair of pickle files (signal and pedestal) the calculates the gain
and returns plots showing the maximum voltages vs time, the charge spectrum, 
the signal amplitude spectrum and a series of oscilloscope traces.
HOW TO RUN IT?
The syntax is the same as for picke_data_lcwp.py, namely:
python make_spectra.py <directory> <channel>
Where <directory> is now the name that has been assigned to the output of the
pickle files produced previously. These are required to also be in a folder
called 'data', but will be placed here automatically when produced. This code
requires another folder called 'plots' to be placed in the current working
directory as this is where it will place its output. The gain will be printed
to terminal.
PARAMETERS TO CONSIDER?
n_samples has to be copied over to match that in the pickle file.
lower_time and upper_time can sometimes cause an issue where no signals are 
detected - say if you move the delay on the 'scope. The max voltage spectrum 
can be useful for spotting where these should be set - look at where the bulk
of signal events occur.
voltage_threshold has a strong effect on the 'Time Over Threshold (TOT)' gain
method. Play with this to find what gives the best green histogram on the
charge spectrum plot.
Often the script will fail when it tries to plot the sample traces for weaker
signals. This is becuase it can't find enough to take a sample of four from.
This can be solved by changing the limits of subset1, subset2 and subset3. 
You can also eliminate it entirely, but I wouldn't reccomend this as it's a 
good indicator of the quality of the data.

