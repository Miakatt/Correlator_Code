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
import pyqtgraph.opengl as gl
import numpy as np

from scipy import fftpack, ndimage
import random
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from PIL import Image
import matplotlib.image as mpimg
import time
from timeit import default_timer as timer
#from matplotlib.pylab import imshow, jet, show, ion
from numpy import meshgrid, fft, exp, linspace, angle, array, pi, ceil, floor, lib, delete

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

    output = fft.ifft2(array(PhaseReferenceFT) * exp(-1j * abs(angle(array(PhaseTargetFT)))))
    output = fft.fftshift(output)

    outputIntensity = abs(output)**2
    return outputIntensity
#=============================================================================================

#def FlipFilter(Filter):
#    return flipud(Filter)
#=============================================================================================


def ComputeFilter(InputImage):

    PhaseFilter = exp(1j * pi * InputImage)
    PhaseFilterFT = fft.fft2(PhaseFilter)

    return PhaseFilterFT

#=============================================================================================

def Binarize(Filter):
    return (Filter > 0.0).astype(int)

#=============================================================================================

def EmbedTarget(Reference, targetSize, target):
    # Check if target dimensions are odd. Subtract row or column to make the dimension even.
    # otherwise, the padded target will be +/- 1 compared to the reference.

    if targetSize[0]%2 != 0:
        target = delete(target, (0), axis=0)
    if targetSize[1]%2 != 0:
        target = delete(target, (0), axis=1)
    # target size = [row, cols] = [y, x]

    targetX = targetSize[1]
    targetY = targetSize[0]
    if _verbose_:
        print ('Target Size ', targetSize)
        print ('target X', targetX)
        print ('target Y', targetY)
        print ('Reference X: ', len(Reference[0]))
        print ('Reference Y: ', len(Reference))
    PadX = int(ceil(0.5*(len(Reference[0])-targetX)))
    PadY = int(ceil(0.5*(len(Reference)-targetY)))
    if _verbose_:
        print ('PadX  ', PadX)
        print ('PadY  ', PadY)
    resizedTarget = lib.pad(target, ((PadY, PadY),(PadX,PadX)), 'constant', constant_values=(0,0) )
    target = resizedTarget

    return target

#==============================================================================================

def OpenPNG(filename):
    file_data = Image.open(filename)
    file_data = file_data.convert('1')

    InputPNG = mpimg.imread(filename)

    return InputPNG, array(file_data)

#=============================================================================================

def CenterBlock(targetFT):
    P = 10
    centerX = int(floor(0.5 * len(targetFT)))
    centerY = int(floor(0.5 * len(targetFT[0])))
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
    print ('Useage: (1) Pass reference and filter pngs at the command line or \n')
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

outputIntensity = Correlate(PhaseTargetFT, PhaseReferenceFT)
print ("HERE")

import pyqtgraph as pg

pg.mkQApp()
import pyqtgraph.opengl as gl

view = gl.GLViewWidget()

xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
view.addItem(xgrid)
view.addItem(ygrid)
view.addItem(zgrid)

xx = linspace(0, 1, len(Reference[0]))
yy = linspace(0, 1, len(Reference))
Z = outputIntensity
X = xx
Y = yy
print (xx.shape, yy.shape, X.shape)
Surf2 = gl.GLSurfacePlotItem(Y, X, Z)
view.addItem(Surf2)
view.show()
pg.plot()
e = timer()
print (e-s)
print ('Done')

if __name__ == '__main__':

    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
