#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage



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

    ax1 = plt.gca()
    ax1.cla()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.imshow(Ref, cmap='gist_gray', vmin=0, vmax=1)
    plt.draw()
    plt.pause(1)
    plt.imsave('Snippet_'+str(snippet.shape[1])+'x'+str(snippet.shape[0])+'.png', Ref, cmap=plt.cm.gray, vmin=0, vmax = 1)
    plt.clf()

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
    targetFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(paddedtarget)))
    conjugatePhaseTargetFT = -(np.angle(targetFT))
    binaryConjugateTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)


    return binaryConjugateTargetFT

#oooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOoooooooooooooooooooooooooo


def ApplyDCBlock(filter, dc):

    Center = [np.floor(0.5 * filter.shape[0]), np.floor(0.5 * filter.shape[1])]
    # Convert Center to integers
    Center = [int(c) for c in Center]
    filter[Center[0]-dc:Center[0]+dc, Center[1]-dc:Center[1]+dc] = 1
    return filter

#oooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOoooooooooooooooooooooooooo

def InitializeReference(RefSize):
    Ref = np.zeros(RefSize)
    return Ref



#========================================================================================================================
#============================================ CODE STARTS HERE ==========================================================
#========================================================================================================================
# Make some variables

# Shape of the snippets (Doesn't have to be square!)
DataShape = ([16,16], [32,32], [16, 64], [16, 128], [100, 100])
# Shape of the input image (size of the SLM)
RefSize = [1100 , 2000]
# spacing between snippet images when tiled on the input image
Spacing = 100

# Size of the filter image (FourierFilter.png)
filtersize = 1000

DCBlock = [5,10,20,50]


#========================================================================================================================

for ds in DataShape:
    Ref = InitializeReference(RefSize)
    rawData = GenerateRandomData(ds)
    Ref = MakeReferencePage(Ref, rawData, Spacing)
    filter = ComputeFilter(rawData, filtersize)
    plt.imsave('Snippet_'+str(ds[1])+'x'+str(ds[0])+'_No_DCBlock.png', filter, cmap=plt.cm.gray, vmin=0., vmax=1.0)

    for dc in DCBlock:
        filter = ApplyDCBlock(filter, dc)
        plt.imsave('Snippet_'+str(ds[1])+'x'+str(ds[0])+'_DCBlock_'+str(dc)+'x'+str(dc)+'.png', filter, cmap=plt.cm.gray, vmin=0., vmax=1.0)


#========================================================================================================================