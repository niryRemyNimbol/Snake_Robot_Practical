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
import brian2
import matplotlib.pyplot as plt
import display


# Clean up all previous communication threads
vrep.simxFinish(-1) 

# Start a new communication thread
clientID=vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID!=-1:
    print('Connected to remote API server.\n')
else:
    print('Connection failed!\n')
    sys.exit('Connection failed!\n')

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

# Get events from DVS camera
errorCode, signalValue = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_streaming)
time.sleep(1)

errorCode, dvsSignal = vrep.simxReadStringStream(clientID, 'dvsData', vrep.simx_opmode_buffer)
dvsSignalArray = vrep.simxUnpackInts(dvsSignal)
dvsEventsList = numpy.reshape(dvsSignalArray, [int(len(dvsSignalArray)/4), 4]).T

# Create the spiking neural network for circle detection and the event generator and link them
nn = snn.snn(32, 32, 16)
events, times, indices = snn.events_generator(nn, dvsEventsList)
s = snn.link_event_to_snn(events, nn)

# Aquire the event outputs as well as the potential evolution in the network
#M = brian2.StateMonitor(nn, 'v', record=True)
#SpikeM1 = brian2.SpikeMonitor(events)
SpikeM2 = brian2.SpikeMonitor(nn)
network = brian2.Network(brian2.collect())
#network.add(M, SpikeM1, SpikeM2)
network.add(SpikeM2)

# Run the network
network.run(1000*brian2.ms)
spikes_n = numpy.array(SpikeM2.i)
#spikes_e = numpy.array(SpikeM1.i)
t = numpy.array(SpikeM2.t)