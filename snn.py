#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:10:01 2017

@author: niry
"""

import brian2
import numpy

def snn(xmax, ymax, rmax):
    N = xmax * ymax * rmax
    tau = 500 * brian2.ms
    v0 = 1 * brian2.mvolt
    vth = 10*v0
    eqs = '''
    dv/dt = -(v0/tau)*sign(v) : volt (unless refractory)
    x : 1
    y : 1
    r : 1
    v0 : volt
    tau : second
    vth : volt
    '''
    condition = 'v > vth'
    update = 'v = 0*v0'
    duration = 10 * tau
    
    snn = brian2.NeuronGroup(N, eqs, threshold=condition, reset=update, refractory = duration, method='euler')
    snn.x = numpy.arange(0, xmax, 1).repeat(ymax*rmax)
    snn.y = numpy.arange(0, ymax, 1).repeat(xmax*rmax).reshape((xmax*rmax, ymax)).flatten('F')
    snn.r = numpy.arange(1, rmax+1, 1).repeat(xmax*ymax).reshape(rmax, xmax*ymax).flatten('F')
    L = len(snn.x)
    snn.v0 = v0 * numpy.ones((1,L))
    snn.tau = tau * numpy.ones((1,L))
    snn.vth = vth * numpy.ones((1,L))
    
    return snn

def events_generator(snn, dvsEventsList):
    nb_events = len(dvsEventsList[0])
    xmax = int(numpy.max(snn.x) + 1)
    ymax = int(numpy.max(snn.y) + 1)

    times = dvsEventsList[0] * brian2.ms
    
    indices = dvsEventsList[1] + ymax * dvsEventsList[2]
    
    N = xmax * ymax
    
    return brian2.SpikeGeneratorGroup(N, indices, times), times, indices
    
def link_event_to_snn(events, snn):
    rmax = int(numpy.max(snn.r))    
    xmax = int(numpy.max(snn.x) + 1)
    ymax = int(numpy.max(snn.y) + 1)

    v_update = 0.5 * brian2.mvolt
    
    synapses = brian2.Synapses(events, snn, model='v_update : volt', on_pre='v += v_update')
    synapses.connect(j='k for k in range(i*rmax, (i+1)*rmax, 1)')
    synapses.v_update = v_update * numpy.ones((1, xmax * ymax * rmax))
    
    return synapses