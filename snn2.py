#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:38:46 2017

@author: niry
"""
import brian2
import numpy
import snn

class SNN(object):
    def __init__(self, xmax, ymax, rmax, dvsSignal):
        self.xmax = xmax
        self.ymax = ymax
        self.rmax = rmax
        self.dvsSignal = dvsSignal
        self.v_update = 0.5 * brian2.mvolt
        
        self.group = snn.snn(self.xmax, self.ymax, self.rmax)
        self.network_op =  brian2.NetworkOperation(self.update_func, dt=100*brian2.ms)
        #self.synapses = snn.inhibition(self.group)
        self.spikeM = brian2.SpikeMonitor(self.group)
        self.network = brian2.Network(self.group, self.network_op, self.spikeM)#, self.synapses)
        
    def run(self, runtime):
        self.network.run(runtime)
        
    def update_func(self):
        time = int(brian2.defaultclock.t/brian2.ms)
        print(time)
        
        M, N = self.dvsSignal.shape
        events = self.dvsSignal[((self.dvsSignal[0,:] <= time) * (self.dvsSignal[0,:] > time - 100)).repeat(4).reshape((N, 4)).T]
        events = events.reshape((4, int(len(events)/4)))
        #if(time>0):
         #   print(events[:,0])
        
        for e in events.T:
            x0 = e[1]
            y0 = e[2]
            for r in range(4, self.rmax, 4):
                if(x0+r<self.xmax and x0-r>=0 and y0+r<self.ymax and y0-r>=0):
                    x, y = snn.solve_centers(x0, y0, r, self.xmax, self.ymax)
                    self.group.v[(r//4-1)*self.xmax*self.ymax+x*self.ymax+y] += self.v_update
        fire = int(numpy.argmax(self.group.v))
        print(self.group.v[fire]/brian2.mvolt, fire)
        if self.group.v[fire]/brian2.mvolt >= 10.0:
            self.group.v = 0 * self.group.v
            self.group.v[fire] = 501*brian2.mvolt
        
    
    def spikes(self):
        spikes = numpy.array(self.spikeM.i)
        times = numpy.array(self.spikeM.t)
        
        return times, spikes
    
    def update_dvsSignal(self, dvsSignal):
        self.dvsSignal = dvsSignal
        