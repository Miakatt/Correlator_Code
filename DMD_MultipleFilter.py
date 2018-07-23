#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage
from math import cos, radians
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
    plt.imsave('input_DMD_rotated'+str(n)+'.png',image, cmap = 'gray', vmin=0, vmax=1 )
    return image


#============================================================================================

# Makes input from of multiple input images.
def MakeFrame(DMDSize, target,N, n):
    frame = np.zeros(DMDSize)
    print(DMDSize[0], DMDSize[1])
    cols = (np.arange(N*0.5*targetsize[0]+1, DMDSize[0]-1.1*N*targetsize[0], 2 * N*targetsize[0]))
    rows = (np.arange(N*0.5*targetsize[1]+1, DMDSize[1]-1.1*N*targetsize[1], 2 * N*targetsize[1]))

    for r, row in enumerate(rows):
        for s, col in enumerate(cols):
            frame[int(col):int(col)+int(N*targetsize[0]),int(row):int(row+N*targetsize[1])] = target

    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
    plt.draw()

    plt.imsave('input_DMD_'+str(n)+'.png', frame, cmap='gray', vmin=0, vmax = 1)
    plt.cla()
    print("Saving frame.")
    return frame


#============================================================================================
# Compute the phase filter (uncomment line 74 if binary phase filter is required - i.e. for the 4DD)
def ComputeFilter(target, filtersize, n, rotation):

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
    #binaryConjugatePhaseTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)

    FourierFilter = np.rot90(conjugatePhaseTargetFT, 3)
    FourierFilter = ndimage.interpolation.rotate(FourierFilter, rotation, mode='constant', cval=1, reshape=True)
    plt.imsave('FourierFilter_'+str(n)+'.png', FourierFilter, cmap=plt.cm.gray, vmin=0., vmax=1.0)

    return FourierFilter

#============================================================================================

# Creates a randomized square target image
def MakeTarget(targetsize):

    target = np.ones((targetsize[1], targetsize[0]))
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
    halfwidth = int(np.ceil(filter.shape[1] / 2))
    halfheight = int(np.ceil(filter.shape[0] / 2))

    print('framex1  ', framex1)
    print('framey1  ', framey1)
    print('halfheight  ', halfheight)
    print('halfwidth  ', halfwidth)

    if ( framex1 + halfwidth > framesize[0]):
        framecut = framesize[0] - framex1
        filterframe[framey1 - halfheight: framey1 + halfheight, framex1 - halfwidth: framex1 + framecut] = filter[:,0:halfwidth+framecut]


    elif (framex1 - halfwidth < 0):
        framecut = framex1
        filterframe[framey1 - halfheight: framey1 + halfheight, framex1 - framecut: framex1 + halfwidth] = filter[:,halfwidth-framecut:filter.shape[1]]

    elif (framey1 - halfheight < 0):
        framecut=framey1
        filterframe[framey1 - framecut: framey1 + halfheight, framex1 - halfwidth+1: framex1 + halfwidth] = filter[halfheight-framecut-1 : filter.shape[0] ,:]

    elif (framey1+halfheight > framesize[1]):
        framecut = framesize[1] - framey1
        filterframe[framey1 - halfheight   : framey1+framecut, framex1 - halfwidth + 1: framex1 + halfwidth] = filter[0:halfheight+framecut ,:]


    else:
        print(filter.shape)
        print("gttftt", framey1 - halfheight, framey1 + halfheight, framex1 - halfwidth, framex1 + halfwidth)
        filterframe[framey1 - halfheight: framey1 + halfheight, framex1 - halfwidth: framex1 + halfwidth] = filter


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
    newy = targetsize[1]*cos(radians(angle))
    dimensions = '%.1fx50!' % (newy)
    print (dimensions)
    cmd = ['convert', 'target.png', '-resize', dimensions, 'resized.png']
    print(cmd)
    subprocess.call(cmd, shell=False)

    squidgedTarget = misc.imread('resized.png','L')
    return squidgedTarget

#============================================================================================
#============================================================================================
#============================================================================================



# Scaling factor
N = 1
# Rotate the filter image on the SLM to undo the physics rotation of the device
# The SLM needs to be rotated to fit the 4 orders on
Rot = -45
# Set angle of reflectance between the laser and the optical axis.
reflectAngle = 24 # degrees
# Optical scaling constant, due to input/filter pixels and fourier lens selection.
OctypusScaling = 520
# Array used in the numpy kroncheker product in 'Scaletarget'
scalingarray = np.ones([N,N])
# Size of the input target image (before scaling and filter computation)
targetsize = [50,50]
# Size of the filter SLM
framesize = [1920,1080]
# Size of the phase filter image
filtersize = int(OctypusScaling/N)
# Size of the DMD
DMDSize = [608,684]
# Coordinates for the 4 filters to coincide with the 4 orders from the DMD
coords1 = [428 , -378]
coords2 = [-309, 355]
coords3 = [-342, -416]
coords4 = [460, 391]

# Create an empty filter frame in which to embed the filters
filterframe = np.zeros([framesize[1], framesize[0]])


# MAKE A TARGET
target1 = MakeTarget(targetsize)
# Uncomment if you want two different input images
#target2 = MakeTarget(targetsize)

# Scale the target (in cases where you want the filter image to be smaller)
Scaled_target1 = Scaletarget(target1, scalingarray)
#Scaled_target2 = Scaletarget(target2, scalingarray)
# MAKE A FRAME FULL OF TARGETS
frame1 = MakeFrame(DMDSize, Scaled_target1, N, 1)
# Uncomment if you want two different frames of input images
#frame2 = MakeFrame(DMDSize, Scaled_target, N, 2)


squidgedTarget = squidgeTarget(target1, reflectAngle)
print('size of squidged target: ', squidgedTarget.shape)

# COMPUTE FILTER OF TARGET
filter1 = ComputeFilter(squidgedTarget, filtersize, 1, Rot)
# Uncomment if you want two different filter images on the filter frame
#filter2 = ComputeFilter(target2, filtersize, 2, Rot)


# EMBED FILTER IN TO FILTERFRAME AT COORDS
# Pass filterN to each one if different filters are required. This just copies the same filter 4 times.
MakeFilterFrame(filter1, framesize, coords1)
MakeFilterFrame(filter1, framesize, coords2)
MakeFilterFrame(filter1, framesize, coords3)
MakeFilterFrame(filter1, framesize, coords4)