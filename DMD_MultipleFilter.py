#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage
from math import cos, radians
from PIL import Image
import subprocess

# Skews the image of the DMD to account for the 2:1 rows:columns in the device, if rotating.
# Not used in this code as it wasn't required 
#============================================================================================

def displayImageOnDMD(image,n):
    #image = np.array(image)
    imageCols = image.shape[1]
    print(imageCols)
    counter = 0
    for col in range(int(0.5*imageCols), imageCols-1, 2):
        image[:,col] = np.roll(image[:,col],counter)
        image[:,col+1] = np.roll(image[:,col+1],counter)

        image[:,imageCols-col-2] = np.roll(image[:,imageCols-col-2], -counter)
        image[:,imageCols-col-3] = np.roll(image[:,imageCols-col-3], -counter)
        counter = counter - 1
    image[:,imageCols-1] = np.roll(image[:,imageCols-1],counter)
    plt.imsave('input_DMD_rotated'+str(n)+'.png',image, 'L', cmap = 'gray', vmin=0, vmax=1 )
    return image


#============================================================================================

# Makes input from of multiple input images.
def MakeFrame(DMDSize, allTargets):
    oversizedDMDSize = [3000,3000]
    if 'oversizedDMDImage' not in locals():
        oversizedDMDImage = np.zeros(oversizedDMDSize)


    print(0, oversizedDMDSize[0] - allTargets.shape[0], allTargets.shape[0])
    cols = np.arange(0, oversizedDMDSize[0]-allTargets.shape[0], allTargets.shape[0]+10)
    rows = np.arange(0, oversizedDMDSize[1]-allTargets.shape[1], allTargets.shape[1]+10)

    print (cols)
    print (rows)


    for r in rows:
        for c in cols:
            oversizedDMDImage[c:c+allTargets.shape[1], r:r+allTargets.shape[0]] = allTargets

#    cols = (np.arange(N*0.5*targetsize[1]+1, oversizedDMDSize[1]-1.1*N*targetsize[1], len(target) * N*targetsize[1]))
#    rows = (np.arange(N*0.5*targetsize[0]+1, oversizedDMDSize[0]-1.1*N*targetsize[0], 2 * N*targetsize[1]))
#
#        for tar in target:
#        print (i)
#        for r, row in enumerate(rows):
#            for s, col in enumerate(cols):
#                oversizedDMDImage[int(col):int(col)+int(N*targetsize[0]), int(row): int(row+N*targetsize[1])] = tar

    # Crop out DMDSize area from oversized DMD
    cropY = (oversizedDMDSize[0] - DMDSize[0]) // 2
    cropX = (oversizedDMDSize[1] - DMDSize[1]) // 2

    print(cropX, cropY)
    frame = oversizedDMDImage[cropY:DMDSize[0]+cropY , cropX:DMDSize[1]+cropX]
    print('Frame Shape  ', frame.shape)
    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
    plt.draw()
    plt.pause(1)

    plt.imsave('input_DMD.png', frame, cmap='gray', vmin=0, vmax = 1)
    plt.cla()
    print("Saving frame.")
    return frame


#============================================================================================
# Compute the phase filter (uncomment line 74 if binary phase filter is required - i.e. for the 4DD)
def ComputeFilter(target, filtersize, n, rotation):

    targetX = target.shape[1]
    targetY = target.shape[0]
    PadX = int(np.floor(0.5 * (filtersize[0] - targetX)))
    PadY = int(np.floor(0.5 * (filtersize[1] - targetY)))
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
    binaryConjugatePhaseTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)
    FourierFilter = binaryConjugatePhaseTargetFT
    #FourierFilter = np.rot90(conjugatePhaseTargetFT, 3)
    FourierFilter = ndimage.interpolation.rotate(FourierFilter, rotation, mode='constant', cval=1, reshape=True)
    plt.imsave('FourierFilter_'+str(n)+'.png', FourierFilter, cmap=plt.cm.gray, vmin=0., vmax=1.0)

    return FourierFilter

#============================================================================================

# Creates a randomized square target image
def MakeTarget(targetsize):

    np.random.seed(Seed)
    target = np.random.randint(2, size = (targetsize[1], targetsize[0]))
    if (0):
        for ii in np.arange(0, targetsize[0], 4):
            print(ii)
            target[ii, :] = 0
            target[:, ii] = 0
 #   target[ int(targetsize[0]/2-5):int(targetsize[0]/2+5), : ] = 0
 #   target[:, int(targetsize[1]/2-5):int(targetsize[1]/2+5)] = 0
    plt.imshow(target, cmap=plt.cm.gray, vmin=0, vmax=1)

    plt.figure(2)
    plt.cla()
    ax2 = plt.gca()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.imsave('target.png', target, cmap=plt.cm.gray, vmin=0, vmax=1)


    return target

#============================================================================================

# Makes a frame of size filtersize. Embeds the phase filter image at the locations given by coords.
# To overlay multiple filters, just call this function each time. It doesn't delete the previous filter frame.
def MakeFilterFrame(filter, framesize, coords):


    x1, y1 = coords

    framex1 = int(np.ceil(framesize[0]/2 + x1))
    framey1 = int(np.ceil(framesize[1]/2 - y1))


    print(framex1, framey1)
    halfwidth = int(np.floor(filter.shape[1] / 2))
    halfheight = int(np.floor(filter.shape[0] / 2))

    print('framex1  ', framex1)
    print('framey1  ', framey1)
    print('halfheight  ', halfheight)
    print('halfwidth  ', halfwidth)

    if ( framex1 + halfwidth > framesize[0]):
        framecut = framesize[0] - framex1
        filterframe[framey1 - halfheight: framey1 + halfheight, framex1 - halfwidth+1: framex1 + framecut] = filter[:,0:halfwidth+framecut]


    elif (framex1 - halfwidth < 0):
        framecut = framex1
        filterframe[framey1 - halfheight: framey1 + halfheight, framex1 - framecut+1: framex1 + halfwidth] = filter[:,halfwidth-framecut:filter.shape[1]]

    elif (framey1 - halfheight < 0):
        framecut=framey1
        filterframe[framey1 - framecut: framey1 + halfheight, framex1 - halfwidth+1: framex1 + halfwidth+1] = filter[halfheight-framecut : filter.shape[0] ,:]

    elif (framey1+halfheight > framesize[1]):
        framecut = framesize[1] - framey1
        filterframe[framey1 - halfheight   : framey1+framecut, framex1 - halfwidth + 1: framex1 + halfwidth+1] = filter[0:halfheight+framecut ,:]


    else:
        print(filter.shape)
        filterframe[framey1 - halfheight : framey1 + halfheight , framex1 - halfwidth+1: framex1 + halfwidth] = filter


    plt.imshow(filterframe, cmap = 'gray', vmin =0, vmax=1)
    plt.draw()
    plt.pause(0.5)

    plt.imsave('filterframe.png', filterframe, cmap=plt.cm.gray, vmin=0., vmax=1.0)

#============================================================================================

# Scale the target by N. This is done AFTER the filter has been calculated, and scales the image to be displayed
# on the input SLM so that the scaling in Octypus can remain as 1 (or 1:2, for the DMD).
def Scaletarget(target, scalingarray):
    return np.kron(target, scalingarray)


#============================================================================================

def squidgeTarget(target, angle):
    newY = int(round(targetsize[1]*cos(radians(angle)), 0))
    print(newY, targetsize[0])
    print (type(target))
    squidgedTarget = misc.imresize(target, ( targetsize[0], newY ), 'bilinear')
    plt.imsave('resized_target.png', squidgedTarget, cmap=plt.cm.gray)

   # dimensions = '%.1fx50!' % (newY)
   # print (dimensions)
   # cmd = ['convert', 'target.png', '-resize', dimensions, 'resized.png']
   # print(cmd)
   # subprocess.call(cmd, shell=False)

    #squidgedTarget = misc.imread('resized.png','L')
    return squidgedTarget

#============================================================================================
#============================================================================================
#============================================================================================

# Random Seed. Set to a number to get 4 indentical target image.
Seed = None
# Scaling factor
N = 1
# Rotate the filter image on the SLM to undo the physics rotation of the device
# The SLM needs to be rotated to fit the 4 orders on
Rot = 0
# Set angle of reflectance between the laser and the optical axis.
reflectAngle = 0 # normally 24 degrees
# Optical scaling constant, due to input/filter pixels and fourier lens selection.
OctypusScaling = [1116, 1116] # [x, y]
# Array used in the numpy kroncheker product in 'Scaletarget'
scalingarray = np.ones([N,N])
# Size of the input target image (before scaling and filter computation)
targetsize = [102 ,102]
# Size of the filter SLM
framesize = [2560, 1600]
# Size of the phase filter image
filtersize = [int(OctypusScaling[0]/N) , int(OctypusScaling[1]/N)]  # [x,y]
# Size of the DMD
DMDSize = [1080, 1920]
# Coordinates for the 4 filters to coincide with the 4 orders from the DMD
coords = ([-443, -614],[674, -629],[-431, 499],[686, 487])

# Create an empty filter frame in which to embed the filters
filterframe = np.zeros([framesize[1], framesize[0]])

frame = np.zeros(DMDSize)
# MAKE A TARGET
targetList = []
Scaled_target = []
filterList = []


for it, co in enumerate(coords):
    targetList.append(MakeTarget(targetsize))
    # Scale the target (in cases where you want the filter image to be smaller)
    Scaled_target.append(Scaletarget(targetList[-1], scalingarray))


    #squidgedTarget = squidgeTarget(target1, reflectAngle)
    #print('size of squidged target: ', squidgedTarget.shape)
    #
    # COMPUTE FILTER OF TARGET
    filterList.append(ComputeFilter(targetList[-1], filtersize, 1, Rot))
    # Uncomment if you want two different filter images on the filter frame


    # EMBED FILTER IN TO FILTERFRAME AT COORDS
    # Pass filterN to each one if different filters are required. This just copies the same filter 4 times.
    #for co in coords:
    MakeFilterFrame(filterList[-1], framesize, co)



#for s in range(len(targetList)):
#    plt.figure(num=s)
#    plt.imshow(targetList[s])
#    plt.draw()
#    plt.pause(0.1)
# MAKE A FRAME FULL OF TARGETS
targetpair1 = np.concatenate((Scaled_target[0], Scaled_target[1]), axis=1)
targetpair2 = np.concatenate((Scaled_target[2], Scaled_target[3]), axis=1)
allTargets = np.concatenate((targetpair1, targetpair2), axis=0)
print('Target Size : ' , allTargets.shape[0])
frame = MakeFrame(DMDSize, allTargets)
