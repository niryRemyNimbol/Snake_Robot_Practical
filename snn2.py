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
        #self.rmin = 2
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
        #print(time)
        
        M, N = self.dvsSignal.shape
        events = self.dvsSignal[((self.dvsSignal[0,:] <= time) * (self.dvsSignal[0,:] > time - 100)).repeat(4).reshape((N, 4)).T]
        events = events.reshape((4, int(len(events)/4)))
        #if(time>0):
         #   print(events[:,0])
        
        for e in events.T:
            x0 = int(e[1])
            y0 = int(e[2])
            for r in range(2, self.rmax, 2):
                if(x0+r<self.xmax or x0-r>0 or y0+r<self.ymax or y0-r>0):
                    x, y = snn.solve_centers(int((self.xmax-1)/2), int((self.ymax-1)/2), r, self.xmax, self.ymax)
                    x = x + (x0 - int((self.xmax-1)/2))
                    y = y + (y0 - int((self.ymax-1)/2))
                    y = y[x>=0]
                    x = x[x>=0]
                    y = y[x<self.xmax]
                    x = x[x<self.xmax]
                    x = x[y>=0]
                    y = y[y>=0]
                    x = x[y<self.ymax]
                    y = y[y<self.ymax]
                    self.group.v[(r//2-1)*self.xmax*self.ymax+x*self.ymax+y] += self.v_update
        fire = int(numpy.argmax(self.group.v))
        #print(self.group.v[fire]/brian2.mvolt, fire)
        if self.group.v[fire]/brian2.mvolt >= 15.0:
            self.group.v = 0 * self.group.v
            self.group.v[fire] = 501*brian2.mvolt
            #self.rmin = 2*(fire//(self.xmax*self.ymax) + 1)
        #else:
         #   self.rmin = 2
        
    
    def spikes(self):
        spikes = numpy.array(self.spikeM.i)
        times = numpy.array(self.spikeM.t)
        
        return times, spikes
    
    def update_dvsSignal(self, dvsSignal):
        self.dvsSignal = dvsSignal
        