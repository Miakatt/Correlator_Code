#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import misc, ndimage













# Compute the phase filter (uncomment line 74 if binary phase filter is required - i.e. for the 4DD)
def ComputeFilter(target, filtersize,  rotation):

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
    conjugateAmplitudeTargetFT = np.absolute(targetFT)
    binaryConjugatePhaseTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)
    binaryConjugateAmpTargetFT   = ((conjugateAmplitudeTargetFT > 0.0)).astype(int)
    #FourierFilter = np.rot90(binaryConjugatePhaseTargetFT, 3)
    FourierPhaseFilter = ndimage.interpolation.rotate(binaryConjugatePhaseTargetFT, rotation, mode='constant', cval=0, reshape=True)
    FourierAmpFilter   = ndimage.interpolation.rotate(binaryConjugateAmpTargetFT, rotation, mode='constant', cval=0,
                                                      reshape=True)
    plt.imsave('FourierPhaseFilter.png', FourierPhaseFilter, cmap=plt.cm.gray, vmin=0., vmax=1.0)
    plt.imsave('FourierAmpFilter.png',   FourierAmpFilter,   cmap=plt.cm.gray, vmin=0., vmax=1.0)

    return FourierPhaseFilter, FourierAmpFilter



def MakeKernel():
    kernel = np.zeros((520,520))
    kernel[260:261,:] = 1

    kernel = ndimage.interpolation.rotate(kernel, -135, mode='constant', cval=0, reshape=True)
    misc.imsave('Kernel.png', kernel)

    return kernel


if (len(sys.argv) > 1):
    targetPNG = sys.argv[1]
    filtersize = int(sys.argv[2])
    rotation = int(sys.argv[3])

    target = misc.imread(targetPNG, 'L')

else:
    target = MakeKernel()
    filtersize = 1000
    rotation = -135

print(type(target))
ComputeFilter(target, filtersize, rotation)