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
    tau = 1 * brian2.ms
    v0 = 1 * brian2.mvolt
    vth = 10*v0
    eqs = '''
    dv/dt = (v0/tau)*sign(v) : volt (unless refractory)
    x : 1
    y : 1
    r : 1
    '''
    condition = 'v > vth'
    update = 'v = 0*v0'
    duration = 10 * tau
    
    snn = brian2.NeuronGroup(N, eqs, threshold=condition, reset=update, refractory = duration, method='euler')
    snn.x = numpy.arange(0, xmax, 1).repeat(ymax*rmax)
    snn.y = numpy.arange(0, ymax, 1).repeat(xmax*rmax).reshape((xmax*rmax, ymax)).flatten('F')
    snn.r = numpy.arange(1, rmax+1, 1).repeat(xmax*ymax).reshape(rmax, xmax*ymax).flatten('F')
    
    return snn

def events_generator(dvsEventsList):