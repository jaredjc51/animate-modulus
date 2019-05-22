# -*- coding: utf-8 -*-
"""
Python script for making a video from degradation/recovery measurements
Author: Jared Carter (jjc407@psu.edu)
Last updated 2016-10-31
version 1.1
  *  FFMPEG is invoked directly, so individual frames are not saved.
  *  Animation module used
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, style
import glob
style.use('video')
framerate = 10  # frames / sec (int)
# List all .csv immittance files
rawfile = glob.glob('.\*.csv')
# Clean up filename
filenames = [x[2:] for x in rawfile]
# Get total number of files
num_files = len(filenames)
# Initalize dictionary to hold all of the csv files in memory
d = {}
# Initialize array for deg/rec times
time = np.zeros(num_files+1, dtype=np.int32)
# Initialize last degradation index
last_deg_index = 0
# Store immittance files in memory and get deg/rec times
for i, filename in enumerate(filenames):
    split_name = filename.split('_')
    for part in split_name:
        try:
            # Save deg/rec time
            time[i] = int(part)
        except ValueError:
            pass
    # Store immittance file in dict
    d[filename] = pd.read_csv(filename, usecols=['# f', 'imagM'])

last_deg_index = 54

# How much faster is the animation than real life
speedup = np.multiply(framerate, np.diff(time))
# Round the speed
speedup = np.around(speedup, -2)
# Change deg to rec discontinuity in speedup
speedup[last_deg_index] = speedup[last_deg_index-1]
# last value in speedup array shouldn't be zero
speedup[-1] = speedup[-2]
speedup = [int(x) for x in speedup]
# Convert dictionary to Panel to make things easier
p = pd.Panel(d)

# Initialize figure
fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111)
ax.axis([2e1, 2e6, 1e-6, 3e-4])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('M\"')
init_plot, = ax.loglog([], [], c='gray')
last_plot, = ax.loglog([], [], c='gray')
curr_plot, = ax.loglog([], [], lw=2)
time_text = ax.annotate('null', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=16)
speed_text = ax.annotate('null2', xy=(0.6, 0.5), xycoords='axes fraction', fontsize=16)


def init():
    '''
    Initialization function so the animation module knows what to update
    with each frame
    '''
    init_plot.set_data([], [])
    last_plot.set_data([], [])
    curr_plot.set_data([], [])
    time_text.set_text('null')
    speed_text.set_text('null2')
    return (init_plot, last_plot, curr_plot, time_text, speed_text)


def data_gen():
    '''Data generator updates the data each time a frame is drawn'''
    # These variables are not directly passed into this function
    global p, filenames, last_deg_index, speedup, num_files
    i = 0
    while i <= num_files:  # Returns here
        print str(i)+' of '+str(num_files)
        data = {}
        data['curr_x'] = p.loc[filenames[i], :, u'# f'].values
        data['curr_y'] = p.loc[filenames[i], :, u'imagM'].values
        data['speed'] = '{}x speed'.format(int(speedup[i]))
        if i == 0:
            # Initial case
            data['init_x'] = np.array([])
            data['init_y'] = np.array([])
            data['last_x'] = np.array([])
            data['last_y'] = np.array([])
            data['time'] = 'Degradation time: {0:06d} s'.format(time[i])
        elif i <= last_deg_index:
            # Degradation
            data['init_x'] = p.loc[filenames[0], :, u'# f'].values
            data['init_y'] = p.loc[filenames[0], :, u'imagM'].values
            data['last_x'] = np.array([])
            data['last_y'] = np.array([])
            data['time'] = 'Degradation time: {0:06d} s'.format(time[i])
        elif i <= num_files:
            # Recovery
            data['init_x'] = p.loc[filenames[0], :, u'# f'].values
            data['init_y'] = p.loc[filenames[0], :, u'imagM'].values
            data['last_x'] = p.loc[filenames[last_deg_index], :,
                                   u'# f'].values
            data['last_y'] = p.loc[filenames[last_deg_index], :,
                                   u'imagM'].values
            data['time'] = 'Recovery time: {0:06d} s'.format(time[i])
        else:
            raise StopIteration
        yield data  # Pauses here until the function is called again
        i += 1


def animate(data):
    '''Set the values in the plot'''
    init_plot.set_data(data['init_x'], data['init_y'])  # Initial
    last_plot.set_data(data['last_x'], data['last_y'])  # Last deg
    curr_plot.set_data(data['curr_x'], data['curr_y'])  # Current meas.
    time_text.set_text(data['time'])  # Time annotation
    speed_text.set_text(data['speed'])  # Speedup annotation
    return (init_plot, last_plot, curr_plot, time_text, speed_text)


# Make the animation
anim = animation.FuncAnimation(fig, animate, data_gen, init_func=init,
                               blit=True, save_count=num_files, interval=1000/framerate)
# Save the animation
anim.save('video.mp4',
          extra_args=['-framerate', str(framerate), '-vcodec', 'libx264',
                      '-r', '30'])
