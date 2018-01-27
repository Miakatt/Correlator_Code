#/usr/bin/env pythonw

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
import sys
import csv
import glob
import os
import re
# ===================================================================
def OpenFile(inputfile):
	f = open (inputfile, "r")

	return f

# ===================================================================
def ReadCSV(filename):

    print (filename)
    with open(filename, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        next(read)
        next(read)
        next(read)
        for row in read:
            t.append(float(row[0]))
            data.append(float(row[1]))
    return data, t
# ===================================================================

#def CreateNoisySine():

#    N = 1000  # number of data points
#    t = np.linspace(0, 4 * np.pi, N)
#    data = 1.06 * np.sin(t + 0.001)  + 0.1*np.random.randn(N)  # create artificial data with noise
#    return data, t

# ===================================================================
def Fit(data, t):
    guess_mean = np.mean(data)
    guess_amp = np.max(data) #3*np.std(data)/(2**0.5)
    guess_phase = 0.5
    Time = np.linspace(0, 3.5*np.pi, len(data))
    print ('Guess Amp: ', guess_amp)
    # we'll use this to plot our first estimate. This might already be good enough for you
    # data_first_guess = guess_std * np.sin(Time + guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0] * np.sin(Time + x[1]) + x[2] - data
    est_std, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_std * np.sin(t + est_phase) + est_mean

    return data_fit#, data_first_guess

# ===================================================================
def GetRMS(data):
    #Peak = np.max(data_fit)
    #RMS = (1/2**(0.5)) * Peak
    Squared = [d**2 for d in data]
    Means = np.mean(Squared)
    RMS = math.sqrt(Means)
    return RMS
# ===================================================================
def GetCurrentAndPower(RMS):
    ThisCurrent = RMS
    ThisPower = ThisCurrent * Voltage
    return ThisCurrent, ThisPower
# ===================================================================
def PlotResults(data):#, fitted_first_guess):
    plt.figure(1)
    plt.cla()
    plt.plot(data, '-')
#    plt.plot(data_fit)
 #   plt.plot(data_first_guess, label='first guess')
    plt.legend()
    plt.draw()
    plt.pause(0.01)
# ===================================================================
def PlotPower(Power, t, sum_Of_t):
   # T = np.linspace(0, sum_Of_t, len(t))
    plt.figure(2)
    plt.cla()
    ax2 = plt.gca()

    plt.plot(sum_Of_t, Power, 'r-x', label='Average Power ($\lambda=62$ms)')
    plt.xlabel('Time (mins)')
    plt.ylabel('Average Power (W)')
    plt.title('InnovateUK Power Test '+ Run)
    plt.legend(loc = 'best')
    ax2.yaxis.set_major_locator(majorLocator)
    ax2.yaxis.set_major_formatter(majorFormatter)
    ax2.yaxis.set_minor_locator(minorLocator)
   # plt.draw()
   # plt.pause(0.001)

# ===================================================================
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# ===================================================================
Run = "Run_1"
F = open("Current_and_Power.txt", 'a')
data = []
t    = []
DataList = []
Current = []
Power   = []
Voltage = 240.0
sum_Of_t = []
sum_Of_t.append(0)
#fig1 = plt.figure(num=1, figsize=(12,8))
plt.figure(num=2, figsize=(12,8))
majorLocator = MultipleLocator(50)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(10)

for infile in sorted(glob.glob('../InnovateUK/'+Run+'/*.csv'), key=numericalSort):
    print ("Current File Being Processed is: " + infile)
    DataList.append(infile)
    print (DataList[-1])

# 1:24:18 run time = 84.3 mins.
# The drop off when the run ends is at csv number 1196
TimePerFile = 84.3/1196

for ii, file in enumerate(DataList):

    data, t = ReadCSV(file)

  # data_fit, data_first_guess = Fit(data, t)
  # data_fit = Fit(data, t)
    yhat = savgol_filter(data, 51, 3)  #
    RMS = GetRMS(yhat)
    print ('RMS : ', RMS)
    ThisCurrent, ThisPower = GetCurrentAndPower(RMS)

    print ('Current %.2f A ' % ThisCurrent)
    print ('Power   %.2f W ' % ThisPower)

    F.write("{0:.3f}".format(sum_Of_t[-1]) + ', ' + "{0:.3f}".format(ThisCurrent) +', '+ "{0:.3f}".format(ThisPower)+'\n')

    Current.append(ThisCurrent)
    Power.append(ThisPower)




    #PlotResults(yhat) #, data_first_guess)
    if (ii>4000):
        break

    sum_Of_t.append(sum_Of_t[-1] + TimePerFile)
    print (sum_Of_t[-1])

    del data[:]
    del t[:]

PlotPower(Power, t, sum_Of_t)
plt.savefig('InnovateUK_Results_'+Run+'.png')
plt.show()
print ('Done')
