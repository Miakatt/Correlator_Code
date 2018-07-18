#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage

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



# input for dmd
def MakeFrame(SLMSize, target,N, n):
    frame = np.ones(SLMSize)
    print(SLMSize[0], SLMSize[1])
    cols = (np.arange(N*0.5*targetsize[0]+1, SLMSize[0]-1.1*N*targetsize[0], 2 * N*targetsize[0]))
    rows = (np.arange(N*0.5*targetsize[1]+1, SLMSize[1]-1.1*N*targetsize[1], 2 * N*targetsize[1]))

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



def MakeTarget(targetsize):

    target = np.ones((targetsize[1], targetsize[0]))
    target = np.random.randint(2, size = (targetsize[1], targetsize[0]))
    #target[ int(targetsize[0]/2-5):int(targetsize[0]/2+5), : ] = 0
    #target[:, int(targetsize[1]/2-5):int(targetsize[1]/2+5)] = 0
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


def Scaletarget(target, scalingarray):
    return np.kron(target, scalingarray)



N = 1
OctypusScaling = 520
scalingarray = np.ones([N,N])
targetsize = [50,50]
framesize = [1920,1080]
filtersize = int(OctypusScaling/N)
SLMSize = [608,684]
coords1 = [416 , -382]
coords2 = [-327,353]
coords3 = [-320, -385]
coords4 = [411, 355]
filterframe = np.zeros([framesize[1], framesize[0]])

# MAKE A TARGET
target1 = MakeTarget(targetsize)
#target2 = MakeTarget(targetsize)
Scaled_target = Scaletarget(target1, scalingarray)
plt.imsave('ScaledTarget.png', Scaled_target, cmap = 'gray', vmin=0, vmax=1)

# MAKE A FRAME FULL OF TARGETS
frame1 = MakeFrame(SLMSize, Scaled_target, N, 1)

#image = displayImageOnDMD(frame1, 1)

#frame2 = MakeFrame(SLMSize, target2, 2)

# COMPUTE FILTER OF TARGET
filter1 = ComputeFilter(target1, filtersize, 1, -45)
#filter2 = ComputeFilter(target2, filtersize, 2)
# EMBED FILTER IN TO FILTERFRAME AT COORDS

MakeFilterFrame(filter1, framesize, coords1)
MakeFilterFrame(filter1, framesize, coords2)
MakeFilterFrame(filter1, framesize, coords3)
MakeFilterFrame(filter1, framesize, coords4)