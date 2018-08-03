#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage
from math import cos, radians
from PIL import Image
import subprocess


# Makes snippets (Small NxM random patterns) tiled across the Input SLM and the corresponding Binary Filter



# Create random snippet of size DataShape
def GenerateRandomData(DataShape):
	# Generate random data with varied saturation.
    rawData = np.random.choice([0,1], size=DataShape, p=[0.5,0.5])
    return rawData

#oooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOoooooooooooooooooooooooooo


# Tile snippets
def MakeReferencePage(Ref, snippet, Spacing):


    for y in np.arange(0, Ref.shape[0] - snippet.shape[0], snippet.shape[0]+Spacing):
        for x in np.arange(0, Ref.shape[1] - snippet.shape[1], snippet.shape[1] + Spacing):

            Ref[y : y+snippet.shape[0], x : x+snippet.shape[1]] = snippet

    plt.figure(1)
    plt.cla()
    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.imshow(Ref, cmap='gist_gray', vmin=0, vmax=1)
    plt.draw()
    plt.pause(1)
    plt.imsave('InputSLM.png', Ref, cmap=plt.cm.gray, vmin=0, vmax = 1)

    return Ref


#oooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOoooooooooooooooooooooooooo


# Compute the phase filter (uncomment line 74 if binary phase filter is required - i.e. for the 4DD)
def ComputeFilter(target, filtersize):

    targetX = target.shape[1]
    targetY = target.shape[0]
    PadX = int(np.floor(0.5 * (filtersize - targetX)))
    PadY = int(np.floor(0.5 * (filtersize - targetY)))
    print('targetX %2.1f : targetY %2.1f  : PadX  %2.1f : PadY %2.1f ' % (targetX, targetY, PadX, PadY))

    paddedtarget = np.pad(target, [(PadY, ), (PadX, ) ], mode='constant', constant_values = 1.0)
    plt.figure(3)
    ax3 = plt.gca()
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.imsave('PaddedTarget.png', paddedtarget, cmap=plt.cm.gray, vmin=0., vmax=1.0)

    targetFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(paddedtarget)))
    conjugatePhaseTargetFT = -(np.angle(targetFT))
    binaryConjugateTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)

    plt.imsave('FourierFilter.png', binaryConjugateTargetFT, cmap=plt.cm.gray, vmin=0., vmax=1.0)

    return binaryConjugateTargetFT

#oooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOoooooooooooooooooooooooooo




# Make some variables

# Shape of the snippets (Doesn't have to be square!)
DataShape = [32,32]
# Shape of the input image (size of the SLM)
Ref = np.zeros((1080, 1920))

# spacing between snippet images when tiled on the input image
Spacing = 100

# Size of the filter image (FourierFilter.png)
filtersize = 1000




rawData = GenerateRandomData(DataShape)
Ref = MakeReferencePage(Ref, rawData, Spacing)
ComputeFilter(rawData, filtersize)