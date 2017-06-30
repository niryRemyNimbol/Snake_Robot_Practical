#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:10:01 2017

@author: niry
"""
# Contains the functions for the circle hough transform spiking neural network

import brian2
import numpy

# Function to create the neurons
# takes as input the length and height of the dvs image (xmax, ymax) and the maximal radius (rmax)
# neurons are in a list: to access the neuron (x, y, r) one calls the (rmax*x + rmax*xmax*y + r - 1)th neuron

def snn(xmax, ymax, rmax):
    # Neuron parameters
    N = xmax * ymax * rmax #size
    # Equation parameters
    tau = 200 * brian2.ms 
    v0 = 1 * brian2.mvolt
    vth = 50*v0 #potential threshold
    eqs = '''
    dv/dt = -(v0/tau)*sign(v) : volt (unless refractory)
    x : 1
    y : 1
    r : 1
    v0 : volt
    tau : second
    vth : volt
    ''' #equations
    condition = 'v > vth'
    update = 'v = 0*v0' #reset rule
    duration = 10 * tau #refractory period
    
    # Network creation
    snn = brian2.NeuronGroup(N, eqs, threshold=condition, reset=update, refractory = duration, method='euler')
    # Parameters initiation
    snn.x = numpy.arange(0, xmax, 1).repeat(ymax*rmax)
    snn.y = numpy.arange(0, ymax, 1).repeat(xmax*rmax).reshape((xmax*rmax, ymax)).flatten('F')
    snn.r = numpy.arange(1, rmax+1, 1).repeat(xmax*ymax).reshape(rmax, xmax*ymax).flatten('F')
    L = len(snn.x)
    snn.v0 = v0 * numpy.ones((1,L))
    snn.tau = tau * numpy.ones((1,L))
    snn.vth = vth * numpy.ones((1,L))
    
    return snn
    
# Function to simulate the event flow of the DVS camera
# takes as parameters a DVS event list and the neural network that will be fed with those events

def events_generator(snn, dvsEventsList):
    xmax = int(numpy.max(snn.x) + 1)
    ymax = int(numpy.max(snn.y) + 1)

    times = dvsEventsList[0] * brian2.ms
    
    indices = dvsEventsList[1] + xmax * dvsEventsList[2]
    
    N = xmax * ymax
    
    return brian2.SpikeGeneratorGroup(N, indices, times), times, indices

# Function to link the event generator with the neural network
# for each event coordinates, computes all the possible centers
   
def link_event_to_snn(events, snn):
    rmax = int(numpy.max(snn.r))    
    xmax = int(numpy.max(snn.x) + 1)
    ymax = int(numpy.max(snn.y) + 1)
    v_update = 0.2 * brian2.mvolt

    indices = events.indices[:]
    xc = indices%xmax
    yc = indices//xmax
    
    synapses = brian2.Synapses(events, snn, model='v_update : volt', on_pre='v += v_update')
    
    for index in range(len(indices)):
        for r in range(1, rmax+1, 1):
            x0 = xc[index]
            y0 = yc[index]
            if(x0+r<xmax and x0-r>=0 and y0+r<ymax and y0-r>=0):
                x, y = solve_centers(x0 ,y0 , r)
                synapses.connect(i=index, j=rmax*x+xmax*rmax*y+r-1)
                #print(x, y, xc[index], yc[index], r)
    N = len(synapses)
    synapses.v_update = v_update * numpy.ones((1, N))
    
    return synapses
    
# Function to compute all the possible center from one event (xc,yc) coordinates and for a given radius r

def solve_centers(xc, yc, r):
    x = numpy.array([xc+r], dtype=int)
    y = numpy.array([yc], dtype=int)
    
    next_px = x[-1] + 1
    next_py = y[-1]
    C = int(numpy.ceil(2*numpy.pi*r))
    for k in range(C):
        x = numpy.append(x, next_px)
        y = numpy.append(y, next_py)
        last_x = x[-1]
        last_y = y[-1]
        next_x = numpy.array([last_x, last_x + 1, last_x + 1, last_x + 1, last_x, last_x - 1, last_x - 1, last_x - 1])
        next_x = (next_x >= 0) * next_x
        next_y = numpy.array([last_y + 1, last_y + 1, last_y, last_y - 1, last_y - 1, last_y - 1, last_y, last_y + 1])
        next_y = (next_y >= 0) * next_y
        i_next = numpy.argmin(abs((next_x - xc)**2 + (next_y - yc)**2 - r**2))
        next_px = next_x[i_next]
        next_py = next_y[i_next]
    return x, y
        