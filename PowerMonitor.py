#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed 3 January 2018

@author: alan.bell@optalysys.com

Record current from Picotech Current Probe. Convert to Power and plot, save data to csv file.

Setup:
1. Connect the SIGLENT Oscilloscope to USB.
2. Connect the Picotech TA018 to to scope
3. Place the TA018 current probe around the positive mains lead only.
# In the Settings (Below)
4. If current probe on 20A setting, set ConversionFactor = 0.1
5. If current probe on 60A setting, set ConverstionFactor = 0.01
6. Provide required file name for .csv file
"""

import sys
import os
from time import strftime
import timeit
import matplotlib.backend_bases
import serial
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import visa
import csv
import gc
from pathlib import Path


#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo



def GetACAmplitude():

    # Optional Function

    AmplString = inst.query("C1: PAVA? AMPL")
    AmplString = AmplString[:-2]
    amplitude  = AmplString.split(',',1)[1:]
    if (verbose):

        print("Channel 1 Amplitude : %0.2f" % 0.5 * float(amplitude[0]))
    return 0.5*float(amplitude[0])

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def GetDCAmplitude():

    # Optional Function

    AmplString = inst.query("C1: PAVA? MEAN")
    AmplString = AmplString[:-2]
    amplitude  = AmplString.split(',',1)[1:]
    if (verbose):

        print("Channel 1 Amplitude : %0.2f" % float(amplitude[0]))
    return float(amplitude[0])

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def GetRMS():
    # Only acts on the periodic part of the waveform.
    # If such a thing doesn't exist, Scope returns ****
    # Use is_number(s) function to convert * to 0 if required.
    RMSString = inst.query("C1: PAVA? CRMS")
    RMSString = RMSString[:-2]
    RMS = RMSString.rsplit(',',1)[1]
    if (verbose):
        print("Channel 1 RMS : %s" % RMS)
    if is_number(RMS):
        return float(RMS)
    else:
        return 0

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def is_number(s):
    # Check if value returned by GetRMS is a number or a '*'
    try:
        float(s)
        return True
    except ValueError:
        return False

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def CalculatePower(RMS):
    Current = RMS/ConversionFactor
    print ('Current  : ', Current)
    Power = Current * SupplyVoltage

    return Power

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def PlotData(RMSList, TimeList):
    plt.figure(1)
    plt.cla()
    plt.plot(TimeList, RMSList, 'r.-', label='Power')
    plt.xlabel("Time (sec)")
    plt.ylabel("Power (Watts)")
    plt.legend(loc = 'best')
    plt.draw()
    plt.pause(0.001)
    plt.savefig(CSVOutput[:-3]+'png', format='png')

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

def WriteToFile(PowerList, TimeList):
    Report = open(CSVOutput, 'a')
    Report.write("%0.2f, %0.3f \n" % (TimeList[-1], PowerList[-1] ) )
    Report.close()

#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo
#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo
#                            SETTINGS
CSVOutput = 'Picotech-Power_Meter.csv'
ConversionFactor = 0.10  # 1mV/10mA
Wait = 1 # How long to wait between measurements
PowerList = []
TimeList = []
verbose = False
SIGLENT_Port = "USB0::0xF4EC::0xEE3A::SDS00001140140::INSTR"
SupplyVoltage = 2.38
#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo
#ooooooooOOOOOOOOOOOOOOOOoooooooooooooOOOOOOOOOOOOOOOooooooooooooooooo

try:
    my_file = Path(CSVOutput)
    if my_file.is_file():
        delete = input('Do you want to remove the previous csv file %s ? (y/n)   ' % CSVOutput)
        if delete == 'y' or delete == 'Y':
            os.remove(CSVOutput)

except OSError:
    pass
SupplyVoltageStr = input('Enter the supply voltage (V)  ')
SupplyVoltage = float(SupplyVoltageStr)

rm = visa.ResourceManager()
inst = rm.open_resource(SIGLENT_Port)
Model = inst.query("*IDN?")
print ("Model: ", Model)

fig1 = plt.figure(num=1, figsize=(8,6), facecolor='white')
tic = timeit.default_timer()
while (1):
    amplitude  = GetDCAmplitude()
    #RMS = 0.707 * amplitude
    RMS = amplitude
    #RMS = GetRMS()
    print('Amplitude : ', amplitude)
  #  print('RMS       : ', RMS)

    Power = CalculatePower(RMS)
    print ('Power    :', Power)
    PowerList.append(Power)
    print ('-----------------')
    TimeList.append(timeit.default_timer() - tic)

    WriteToFile(PowerList, TimeList)

    PlotData(PowerList, TimeList)

    time.sleep(Wait)


inst.close()



