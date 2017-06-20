#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:41:21 2017

@author: niry
"""


import brian2
import matplotlib.pyplot as plt

eq2 = '''
dv/dt = -sign(v)*v0/tau0 : volt (unless refractory)
'''
eq1='''
dv/dt=v0/tau : volt (unless refractory)
'''
v0 = 1 * brian2.mvolt
tau = 1*brian2.second
tau0 = 5*brian2.second
neuron1 = brian2.NeuronGroup(1, eq1, threshold='v>0.01*v0', reset='v=0*v0', refractory=15*brian2.ms, method='euler')
neuron2 = brian2.NeuronGroup(1, eq2, threshold='abs(v)>0.02*v0', reset='v=0*v0', refractory=20*brian2.ms, method='euler')
#S = brian2.Synapses(neuron1, neuron2, on_pre='v_post+=0.01*v0')
S = brian2.Synapses(neuron1, neuron2, on_pre='v_post-=0.01*v0')
S.connect(j='i')
M1 = brian2.StateMonitor(neuron1, 'v', record=True)
M2 = brian2.StateMonitor(neuron2, 'v', record=True)
network = brian2.Network(brian2.collect())
network.add(M1,M2)
network.run(500*brian2.ms)
plt.plot(M2.t/brian2.ms, M2.v[0])
plt.plot(M1.t/brian2.ms, M1.v[0])
plt.show()