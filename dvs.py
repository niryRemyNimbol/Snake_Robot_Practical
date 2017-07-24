#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:18:38 2017

@author: niry
"""

import vrep
import sys
import time
import numpy
import snn
import snn2
import brian2
import matplotlib.pyplot as plt
import display

# Build network
#nn = snn.snn(64, 64, 32)
print('Building neural network ...\n')
network = snn2.SNN(64,64,32,numpy.array([]))
#time.sleep(5)
# Clean up all previous communication threads
vrep.simxFinish(-1) 

# Start a new communication thread
clientID=vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID!=-1:
    print('Connected to remote API server.\n')
else:
    print('Connection failed!\n')
    sys.exit('Connection failed!\n')
    
# Enable synchronous mode
vrep.simxSynchronous(clientID, True)

# Get handle for the DVS camera
#errorCode, dvsHandle = vrep.simxGetObjectHandle(clientID, 'DVS128_sensor', vrep.simx_opmode_blocking)

#errorCode, resolution, image=vrep.simxGetVisionSensorImage(clientID, dvsHandle, 1, vrep.simx_opmode_streaming)
#errorCode, resolution, image=vrep.simxGetVisionSensorImage(clientID, dvsHandle, 1, vrep.simx_opmode_buffer)
#im = numpy.reshape(image,[128,128])

# To get event : simxReadStringStream + simxUnpackInts
#errorCode, signalValue = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_streaming)
#time.sleep(5)
#errorCode, dvsSignal = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_buffer)
#dvsSignalArray = vrep.simxUnpackInts(dvsSignal)
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
# Get events from DVS camera
errorCode, signalValue = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_streaming)
time.sleep(0.01)

vrep.simxSynchronousTrigger(clientID)
errorCode, dvsSignal = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_buffer)
dvsSignalArray = vrep.simxUnpackInts(dvsSignal)
dvsEventsList = numpy.reshape(dvsSignalArray, [int(len(dvsSignalArray)/4), 4]).T
while(True):
    print('Receiving DVS data ...\n')
    errorCode, dvsSignal = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_buffer)
    dvsSignalArray = vrep.simxUnpackInts(dvsSignal)
    dvsEventsList = numpy.concatenate((dvsEventsList,numpy.reshape(dvsSignalArray, [int(len(dvsSignalArray)/4), 4]).T),1) 

    vrep.simxPauseSimulation(clientID, vrep.simx_opmode_oneshot)
    #print(dvsSignalArray)

# Create the spiking neural network for circle detection and the event generator and link them
#nn = snn.snn(32, 32, 16)
    
    #events, times, indices = snn.events_generator(nn, dvsEventsList)
    #s0 = snn.link_event_to_snn(events, nn)
#s1 = snn.inhibition(nn)

# Aquire the event outputs as well as the potential evolution in the network
#M = brian2.StateMonitor(nn, 'v', record=True)
#SpikeM1 = brian2.SpikeMonitor(events)
    #SpikeM2 = brian2.SpikeMonitor(nn)
    #network = brian2.Network(brian2.collect())
#network.add(M, SpikeM1, SpikeM2)
    #network.add(SpikeM2)

# Run the network
    #network.run(1000*brian2.ms)
    #spikes_n = numpy.array(SpikeM2.i)
#spikes_e = numpy.array(SpikeM1.i)
    #t = numpy.array(SpikeM2.t)

    print('Running network ...\n')
    network.update_dvsSignal(dvsEventsList)
    network.run(149*brian2.ms)

    t, spikes_n = network.spikes()
# Display results
    print('frame at time:', t[-1], ' seconds')
    im=display.reconstruct_image(dvsEventsList,int(1000*t[-1]))
    display.draw_circle(im,spikes_n[t==t[-1]])
    
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    vrep.simxSynchronousTrigger(clientID)

    #time.sleep(1)
    
    #vrep.simxSynchronous()
    #vrep.simxSynchronousTrigger()
    