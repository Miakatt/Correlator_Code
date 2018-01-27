#!/usr/bin/env python

import sys
from scipy import fftpack, ndimage
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from matplotlib.widgets  import RectangleSelector
from PIL import Image
from random import randint


def GenerateSearchField(length):
    # Generate the alphabetical characters of the search field
    Field = []
    for i in range(0, length):
        Field.append(random.choice('TGAC'))

    return Field


#=============================================================================================

def EncodeField(outputField, inputField, EncodeDepth, width):
    # Convert the letters in the search field to black & white encoded images

    for i, val in enumerate(inputField):
        # Get encoded character
        EncodedCharacter = EncodeCharacters(val)
        outputField[EncodeDepth * int(i/width): EncodeDepth * int(i/width)+EncodeDepth , int(i%width) ] = EncodedCharacter

    return outputField


#=============================================================================================

def EncodeCharacters(Character):
    if Character == 'T':
        Encode = [1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1]
    elif Character == 'G':
        Encode = [0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0]
    elif Character == 'A':
        Encode = [1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,0,0,1,1]
    elif Character == 'C':
        Encode = [0,0,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,0,0]
    return Encode

#=============================================================================================
def EmbedTarget(Reference, targetSize, target):

    # target size = [row, cols] = [y, x]
    print ('Target Size ') , targetSize
    M = 1000
    N = 1000
    targetX = targetSize[0]
    targetY = targetSize[1]
    print ('target X'), targetX
    print ('target Y'), targetY
    print ('Reference X: '), len(Reference[0])
    print ('Reference Y: '), len(Reference)
    PadX = int(np.floor(0.5*(M-targetX)))
    PadY = int(np.floor(0.5*(N-targetY)))
    plt.figure(num=2, facecolor='white')
    ax2 = plt.gca()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    resizedTarget = np.lib.pad(target, ((PadX, PadX), (PadY, PadY)), 'constant', constant_values=(0, 0))
    target = resizedTarget
    plt.imshow(target, cmap='Greys')
    plt.draw()
    plt.pause(0.01)
    plt.imsave('target.png', target, cmap=plt.cm.gray_r, vmin=0, vmax = 1)
    return target


#=============================================================================================

def ComputeFilter(target, Threshold):
    # Compute Filter
    targetFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target)))

    plt.draw()
    plt.pause(0.001)

    # print targetFT
    conjugatePhaseTargetFT = -(np.angle(targetFT))

    binaryConjugatePhaseTargetFT = ((conjugatePhaseTargetFT > Threshold)).astype(int)
 #   binaryConjugatePhaseTargetFT = conjugatePhaseTargetFT
    #binaryConjugatePhaseTargetFT = conjugatePhaseTargetFT
    plt.figure(num=3, facecolor='white')
    ax3 = plt.gca()
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.imshow(binaryConjugatePhaseTargetFT, cmap='Greys', vmin=0, vmax=1.0)
  #  plt.savefig('FourierFilter.png', bbox_inches='tight', pad_inches=0)
    plt.draw()
    plt.pause(0.01)
    return binaryConjugatePhaseTargetFT

#=============================================================================================

def ShowMatrix(Matrix):

    plt.figure(num=1, facecolor='white')
    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.imshow(Matrix, cmap='Greys', vmin=0, vmax=1)
    plt.draw()
    plt.pause(0.001)

#=============================================================================================

def InputSnippets(Reference, target, EncodeDepth, width, gapSize, Rows):

    for Y in np.arange(100, Rows, 100):
        for X in np.arange(0, width - len(target[0]), len(target[0])+gapSize ):
            Reference[Y:Y+EncodeDepth, X:X+len(target[0])] = target

    return Reference


#=============================================================================================

def ZeroFilterBlock(BinaryFilter):

    midpointX, midpointY = (0.5*BinaryFilter.shape[0], BinaryFilter.shape[1])

    BinaryFilter[midpointX-0.5*block_size:midpointX+0.5*block_size, midpointY+0.5*block_size:midpointY-0.5*block_size] = 0
    return BinaryFilter

#=============================================================================================


# Define Zero Block size if used. Set to 0 if no block required
block_size = 0

width =  1920
Rows =   1080
length = 64
gapSize = 32
EncodeDepth = 24

Reference = 0.5*np.ones((Rows, width))
target    = np.zeros((EncodeDepth, length))



Field = GenerateSearchField(length)
target = EncodeField(target, Field, EncodeDepth, width)
print (target.shape)

targetSize = target.shape
ShowMatrix(target)

plt.imsave('Pre-Padded_Target.png', target, cmap=plt.cm.gray, vmin=0, vmax=1)


Reference = InputSnippets(Reference, target, EncodeDepth, width, gapSize, Rows)
plt.imshow(Reference, cmap='Greys', vmin=0, vmax=1.0)
plt.imsave('1080_by_1920_Reference.png', Reference, cmap=plt.cm.gray, vmin=0, vmax=1)

target = EmbedTarget(Reference, targetSize, target)
print (target.shape)
BinaryFilter = ComputeFilter(target, 0.0)
if block_size > 0:
    BinaryFilter = ZeroFilterBlock(BinaryFilter)
    plt.figure(3)
    plt.imshow(BinaryFilter, cmap='Greys', interpolation='nearest', vmin=0, vmax=1.0)

plt.imsave('1080_by_1920_Filter_r.png', BinaryFilter, cmap=plt.cm.gray_r, vmin=0, vmax=1)
plt.imsave('1080_by_1920_Filter.png', BinaryFilter, cmap=plt.cm.gray, vmin=0, vmax=1)

print ('Done')
plt.show()