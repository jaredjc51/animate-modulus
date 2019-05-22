# -*- coding: utf-8 -*-
"""
Python script for reading impedance measurements as a function of time
and converting them to all immittance formalisms
Author: Jared Carter, Clive Randall Group
Version 1.2
Last updated: 2018-02-12

v 1.2
  - Improved documentation
  - Rewrote admittance vs. time section to use xarray

v 1.1
  - Added ability to normalize Z and Y to geometry of sample
"""

#%%############################################################################
### Change these values #######################################################

filename = 'B-252_4sweepfreq_deg40_recov.diel'
area = 0.335 * 0.328 # cm^2
thickness = 0.5 # cm
# Save impedance in Z-view compatible format?
save_zview = False
# Area correction? True = Ohm/cm and S/cm for Z and Y
per_cm = True
# Frequencies for admittance vs. time
freq_list = [20, 100]

#%%############################################################################
###############################################################################

# Import modules and define functions

import sys, os # Make folders on Mac and PC without breaking
import numpy as np # Efficient array handling
import xarray as xr # Labeled multi-dimensional arrays
import matplotlib.pyplot as plt # Plotting
e0 = 8.854188e-14 # F/cm (vacuum permittivity)
# Header for exporting immittance data
h = 'f,1/f,realY,imagY,realZ,imagZ,realM,imagM,realE,imagE,tandelta'


def getdata(filename):
    '''
    Function that reads .diel files and returns sweeps and measurements
    '''
    # initialize values
    az = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = 0
    s = {}
    m = {}
    H = True
    # open the file
    with open(filename, 'r+b') as f:
        # iterate over every line in the file
        for line in f:
            # remove extra line breaks
            line = line.replace('\r\n', '')
            # Check if we are in the header
            if H is True:
                # Define a new sweep
                if '_set' in line or 'COUNTER' in line or 'SAMPLE' in line:
                    # Name of the sweep
                    sline = line.split('\t')
                    xname = az[letter]+' '+sline[0]
                    # Values that the sweep uses
                    line = f.next().replace('+', '').replace(' ', '')
                    sline = line.split('\t')
                    del sline[-1]   # Last value in list is empty
                    # Make the strings into numbers
                    xdata = [float(x) for x in sline]
                    # Add to the sweep dictionary
                    s[xname] = xdata
                    letter += 1
                # Make a key in the measurement dictionary
                elif 'REAL' in line or 'CMPLX' in line:
                    xname = line.replace('  ', ' ')
                    m[xname] = []
            # What to do when outside of the header
            elif H is False:
                # This line starts a measurement
                if line in m:
                    key = line
                    try:
                        line = f.next()
                    except StopIteration:
                        break
                    # complex
                    if '\t' in line:
                        sline = line.split('\t')
                        realline = float(sline[0])
                        imagline = float(sline[1])
                        xdata = complex(realline, imagline)
                        m[key].append(xdata)
                    # real
                    else:
                        m[key].append(float(line))
                elif 'LIST_REAL_CMPLX SWEEPFREQ' in line:
                    xname = line
                    rows = int(f.next())
                    m[xname] = np.zeros((rows, 3))
                    for i in range(rows):
                        m[xname][i, :] = f.next().split()
            # Are we in the header?
            if '*' in line:
                H = False
    return s, m


def z_to_y(r, x):
    '''Convert impedance to admittance'''
    den = np.square(r) + np.square(x)
    y1 = np.divide(r, den) 
    y2 = np.divide(x, den) *-1.0
    return y1, y2

def z_to_m(r, x, f, A, t):
    '''Convert impedance to modulus'''
    mu = np.multiply(2*np.pi*e0*A/t, f)
    m1 = np.multiply(mu, x) *-1.0
    m2 = np.multiply(mu, r)
    return m1, m2

def z_to_eps(r, x, f, A, t):
    '''Convert impedance to permittivity'''
    area = A
    thickness = t
    mu = np.divide(
        thickness/(np.add(np.square(r), np.square(x))*2*np.pi*e0*area), f)
    eps1 = np.multiply(mu, x) *-1.0
    eps2 = np.multiply(mu, r)
    return eps1, eps2

def z_to_tand(r,x):
    '''Convert impedance to loss tangent'''
    return r / x

def convert_z(f, r, x, A, t, per_cm=False):
    '''Convert impedance to all immittance formalisms'''
    # Area correction ?
    if per_cm is True:
        geo = (1.0*t)/A
    else:
        geo = 1.0
    # inverse frequency
    inv_f = 1.0 / f
    # Admittance
    y1, y2 = z_to_y(r, x)
    y1 = y1 * geo
    y2 = y2 * geo
    # Impedance
    z1, z2 = (r, x)
    z1 = z1 / geo
    z2 = z2 / geo
    # Modulus
    m1, m2 = z_to_m(r, x, f, A, t)
    # Permittivity
    eps1, eps2 = z_to_eps(r, x, f, A, t)
    # Loss tangent
    tand = z_to_tand(r,x)
    return f, inv_f, y1, y2, z1, z2, m1, m2, eps1, eps2, tand

def find_nearest(array,value):
    '''Find the number in the array closest to the given value'''
    idx = (np.abs(array-value)).argmin()
    return array[idx]

#%% Run the script

# Get path to current folder
my_path = sys.path[0]
# Make immittance file path if it doesn't exist
directory = os.path.join(my_path, 'immittance')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make zview file path if it doesn't exist and we are exporting zview data
if save_zview is True:
    dir2 = os.path.join(my_path, 'zview')
    if not os.path.exists(dir2):
        os.makedirs(dir2)

# Import data
s,m = getdata(filename)
# Get measurement times
times = np.array(m['REAL TIME2'] + m['REAL TIME4'] + m['REAL TIME6']) - m['REAL TIME4'][0] + 1.0
# How many sweeps do we expect
num_sweeps = times.size
# Initial measurement
data0 = m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA 1']
# Frequency
f0 = data0[:,0]
# Real impedance
r0 = data0[:,1]
# Imaginary impedance
x0 = data0[:,2]
# Other immittance formalisms
tup0 = convert_z(f0, r0, x0, area, thickness, per_cm)
# Stack into one array
out0 = np.column_stack(tup0)
# Export immittance
export_name0 = os.path.join(directory, filename[:-5]+'_initial_all.csv')
np.savetxt(export_name0, out0, delimiter=',',header=h)
# Export zveiw
if save_zview is True:
    np.savetxt(os.path.join(dir2, filename[:-5]+'_initial.csv'), data0, delimiter=',')

# Define list to hold admittance data
admittance_list = []
# For every measurement in the Time sweep
for i in range(num_sweeps):
    goodfile = True
    try:
        # First deg measurement ends with 2
        data = m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA {}'.format(i+2)]
    except KeyError:
        goodfile = False
    if goodfile is True:
        # Frequency
        f = data[:, 0]
        # Real impedance
        r = data[:, 1]
        # Imaginary impedance
        x = data[:, 2]
        # Get other immittance formalisms
        tup = convert_z(f, r, x, area, thickness, per_cm)
        admittance_list.append(tup[2])
        # Stack into one array
        out = np.column_stack(tup)
        # Export immittance
        export_name = os.path.join(directory, filename[:-5]+'_{0:05d}_s_all.csv'.format(int(np.around(times[i]))))
        np.savetxt(export_name, out, delimiter=',',header=h)
        # Export zview
        if save_zview is True:
            np.savetxt(os.path.join(dir2, filename[:-5]+'_{0:05d}_s.csv'.format(int(times[i]))), data, delimiter=',')

#%% Admittance vs. time

# Make xarray with dimensions of frequency and time
admittance = xr.DataArray(np.column_stack(admittance_list), coords=[('frequency', f), ('time', times[:-1])])
# Find frequencies closest to specified frequencies
freqs = [find_nearest(f, fff) for fff in freq_list]
# Select only the frequencies wanted from immittance
y_vs_t = admittance.sel(frequency=freqs).T.to_pandas()
# Save the file
y_vs_t.to_csv('admittance_during_deg.csv',index_label='Time (s)')

# Plot it
plt.figure()
for f in freqs:
    plt.semilogx(y_vs_t.index, y_vs_t[f], 'o', label='{:.0e}'.format(f))
plt.legend(loc='best')
plt.title('Real admittance at various frequencies')
plt.xlabel('Time (s)')
plt.ylabel('Real admittance')
plt.show()
