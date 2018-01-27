#!/usr/bin/env python


"""
3/10/2017
For determining the best rotation angle of the 2nd SLM.

Create a 1000x1000 reference image. Take a subset of that (the filter) and get a Fourier Transform.
Flip the filter vertically.
Rotate the filter by angles given in the list 'Angles'.

Save as png's. IMPORTANT: Use imsave function to prevent image scaling seen with savefig(...png)

Example:  python CreateSimpleBinaryFilter.py 255 50 100

"""
import sys

from scipy import fftpack, ndimage
import random

import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from PIL import Image
import matplotlib.image as mpimg
import time
from timeit import default_timer as timer
from matplotlib.pylab import imshow, jet, show, ion
#from numpy import meshgrid, fft, exp, linspace, angle, array, pi, ceil, floor, lib, delete
import numpy as np
from numba import jit

#=============================================================================================

def ShowMatrix(Matrix, N):

    plt.figure(num=N, facecolor='white')
    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.imshow(Matrix, cmap='Greys', vmin=0, vmax=1)
    plt.draw()
    plt.pause(0.1)

#=============================================================================================

def Correlate(PhaseTargetFT, PhaseReferenceFT):

    output = np.fft.ifft2(np.array(PhaseReferenceFT) * np.exp(-1j * abs(np.angle(np.array(PhaseTargetFT)))))
    output = np.fft.fftshift(output)

    outputIntensity = abs(output)**2
    fig = plt.figure(num=20, facecolor='white', figsize=(8,6))
    plt.subplot(111)
  #  plt.imshow(outputIntensity, cmap='hot') #, interpolation='nearest')
  #  plt.colorbar()

  #  plt.subplot(122)
#    fig = plt.figure(num=21, facecolor='white', figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(0, len(Reference[0]),len(Reference[0])), np.linspace(0,len(Reference),len(Reference)))
    Z = outputIntensity
    X = xx
    Y = yy

    ax21 = fig.add_subplot(111, projection='3d')
    ax21.plot_surface(Y,X,Z, cmap='hot', rstride = Stride, cstride = Stride)
    ax21.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax21.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    plt.draw()
    plt.pause(0.001)

#=============================================================================================

#def FlipFilter(Filter):
#    return flipud(Filter)
#=============================================================================================


def ComputeFilter(InputImage):

    PhaseFilter = np.exp(1j * np.pi * InputImage)
    PhaseFilterFT = np.fft.fft2(PhaseFilter)

    return PhaseFilterFT

#=============================================================================================

def Binarize(Filter):
    return (Filter > 0.0).astype(int)

#=============================================================================================

def EmbedTarget(Reference, targetSize, target):
    # Check if target dimensions are odd. Subtract row or column to make the dimension even.
    # otherwise, the padded target will be +/- 1 compared to the reference.

    if targetSize[0]%2 != 0:
        target = np.delete(target, (0), axis=0)
    if targetSize[1]%2 != 0:
        target = np.delete(target, (0), axis=1)
    # target size = [row, cols] = [y, x]

    targetX = targetSize[1]
    targetY = targetSize[0]
    if _verbose_:
        print ('Target Size ', targetSize)
        print ('target X', targetX)
        print ('target Y', targetY)
        print ('Reference X: ', len(Reference[0]))
        print ('Reference Y: ', len(Reference))
    PadX = int(np.ceil(0.5*(len(Reference[0])-targetX)))
    PadY = int(np.ceil(0.5*(len(Reference)-targetY)))
    if _verbose_:
        print ('PadX  ', PadX)
        print ('PadY  ', PadY)
    resizedTarget = np.lib.pad(target, ((PadY, PadY),(PadX,PadX)), 'constant', constant_values=(0,0) )
    target = resizedTarget

    return target

#==============================================================================================

def OpenPNG(filename):
    file_data = Image.open(filename)
    file_data = file_data.convert('1')

    InputPNG = mpimg.imread(filename)

    return InputPNG, np.array(file_data)

#=============================================================================================

def CenterBlock(targetFT):
    P = 10
    centerX = int(np.floor(0.5 * len(targetFT)))
    centerY = int(np.floor(0.5 * len(targetFT[0])))
    targetFT[centerX-P:centerX+P, centerY-P:centerY+P] = 1
    return targetFT
#=============================================================================================




_verbose_ = True

# Detect the number of arguments passed from the command line
# No arguments initiates a dialog window.
s = timer()
if len(sys.argv) == 1:
    print ('No arguments passed... open dialog window')
    # Dialog window syntax has changed between Python 2.x and Python 3.x
    if sys.version_info[0] == 3:
        import matplotlib
        matplotlib.use('TkAgg')  # Required on some macs for drawing dialog windows
        print ('Python 3.x installed')
        import tkinter as tk
        from tkinter import filedialog, constants
        time.sleep(0.1)
        root = tk.Tk()
        root.withdraw()
        print ("SELECT INPUT PNG IMAGE.\n")
        Ref_file_path = filedialog.askopenfilename()
        root.withdraw()
        print ("SELECT TARGET PNG IMAGE.\n")
        Target_file_path = filedialog.askopenfilename()
        root.withdraw()

    elif sys.version_info[0] == 2:
        import matplotlib
        matplotlib.use('TkAgg')  # Required on some macs for drawing dialog windows
        print ('Python 2.x installed')
        import Tkinter, tkFileDialog
        time.sleep(0.1)
        root = Tkinter.Tk()
        root.withdraw()
        print ("SELECT INPUT PNG IMAGE.\n")
        root.update()
        Ref_file_path = tkFileDialog.askopenfilename(title = "Select Input Image")
        root.update()
        print ("SELECT TARGET PNG IMAGE.\n")
        Target_file_path = tkFileDialog.askopenfilename(title = "Select Target Image")
        root.update()
        root.quit()


    # Read in PNGs passed via the command line
    RefPNG, Reference = OpenPNG(Ref_file_path)
    TargetPNG, target = OpenPNG(Target_file_path)

elif len(sys.argv) == 3:
    # Read in PNGs passed via the command line
    RefPNG, Reference = OpenPNG(sys.argv[1])
    TargetPNG, target = OpenPNG(sys.argv[2])
else:
    print ('Useage: (1) Pass reference and target pngs at the command line or \n')
    print ('(2) pass no arguments and select files through the GUI window.\n')

# Sets granularity of the meshplot data. More course = faster, but may miss peaks.
Stride = 5


targetSize = target.shape
target = EmbedTarget(Reference, targetSize, target)
PhaseTargetFT = target
# Option to show the reference or target images. The number it for providing a unique figure window.
#ShowMatrix(Reference, 10)
#ShowMatrix(target, 11)

PhaseReferenceFT    = ComputeFilter(Reference)
PhaseTargetFT       = ComputeFilter(target)
BinaryPhaseTargetFT = Binarize(PhaseTargetFT)
plt.imsave('Binary_Phase_Filter.png', BinaryPhaseTargetFT, cmap=plt.cm.gray, vmin=0, vmax=1)

Correlate(PhaseTargetFT, PhaseReferenceFT)
e = timer()
print (e-s)
print ('Done')
plt.show()
#if len(sys.argv) < 3:

