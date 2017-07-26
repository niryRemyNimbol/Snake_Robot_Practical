#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:22:36 2017

@author: niry
"""

# Contains the functions to display the results of our experiment

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import brian2

# Function to recconstruct the DVS image at time t from the event list
def reconstruct_image(dvsEventsList, t):
    # Get events for time t
    M, N = dvsEventsList.shape
    events = dvsEventsList[(numpy.round(dvsEventsList[0,:]/10)*10 == t).repeat(4).reshape((N, 4)).T]
    events = events.reshape((4, int(len(events)/4)))
    
    # Create gray image
    xmax = 64#max(dvsEventsList[1])+1
    ymax = 64#max(dvsEventsList[2])+1
    im = 127*numpy.ones((xmax,ymax,3))
    im = numpy.uint16(im)
    
    # Add events to the gray image
    for event in events.T:
        if event[3]==1:
            im[event[1],event[2],:]=numpy.ones((1,1,3))
        else:
            im[event[1],event[2],:]=255*numpy.ones((1,1,3))
    
    return im
    
# Function to draw the detected circle on the DVS image
def draw_circle(im, indexes):
    # Compute x, y, r
    xmax, ymax, channel = im.shape
    #rmax = min(xmax, ymax)//2
    centers = []
    for i in indexes:
        r = 4*(i//(xmax*ymax) + 1)
        xy = i%(xmax*ymax)
        x = xy//ymax
        y = xy%ymax
        centers.append((x,y,r))
    # Create a figure. Equal aspect so circles look circular
    #fig,ax = plt.subplots(1)
    #ax.set_aspect('equal')

    # Show the image
    #ax.imshow(numpy.uint16(im))

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for k in centers: 
        circ = Circle((k[1], k[0]), k[2], color='g', fill=False)
        #ax.add_patch(circ) 

    # Show the image
    #plt.show()
    return circ
    
# Function to display firing neurons
def display_spikes(spike_monitor):
    plt.plot(spike_monitor.t/brian2.ms, spike_monitor.i, '+k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    
# Function to display the potential of a neuron over time
def display_potential(state_monitor, index):
    plt.plot(state_monitor.t/brian2.ms, state_monitor.v[index])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')