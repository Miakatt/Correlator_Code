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
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from matplotlib.widgets  import RectangleSelector
from PIL import Image
from random import randint
import CommonFunctions as CF
#=============================================================================================

def GenerateSearchField(length):
    # Generate the alphabetical characters of the search field
    Field = []
    for i in range(0, length):
        Field.append(random.choice('TGAC'))

    return Field
#=============================================================================================

def TestField(length, Reference):
    # Generate a bar pattern for testing the FFT

    for i in range(0, length):
        if i%2 == 0:
            Reference[EncodeDepth * (i/width): EncodeDepth * (i/width)+6 , i%width ] = np.ones((EncodeDepth))
        else:
            Reference[EncodeDepth * (i/width): EncodeDepth * (i/width)+6 , i%width ] = np.zeros((EncodeDepth))
    return Reference


def OpenPNG(filename):
    file_data = Image.open(filename)
    file_data = file_data.convert('1')

    InputPNG = mpimg.imread(filename)

    return InputPNG, np.array(file_data)


#=============================================================================================

def EncodeField(Field):
    # Convert the letters in the search field to black & white encoded images
    for i, val in enumerate(Field):
        # Get encoded character
        EncodedCharacter = EncodeCharacters(val)

        Reference[EncodeDepth * int(i/width): EncodeDepth * int(i/width)+EncodeDepth , int(i%width) ] = EncodedCharacter

    return Reference


#=============================================================================================

def EncodeCharacters(Character):
    if Character == 'T':
        Encode = [1,1,1,0,0,1,0,0,1,0,0,1]
    elif Character == 'G':
        Encode = [0,1,0,0,0,1,1,0,0,1,1,1]
    elif Character == 'A':
        Encode = [0,0,1,0,0,1,0,1,0,1,0,0]
    elif Character == 'C':
        Encode = [0,0,0,0,1,0,0,0,1,1,1,0]
    return Encode

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
   # plt.savefig('Reference.png', bbox_inches='tight', pad_inches=0)
    plt.imsave('Reference.png', Matrix, cmap=plt.cm.gray, vmin=0, vmax = 1)
# plt.pause(0.001)

#=============================================================================================
def EmbedTarget(Reference, targetSize, target):


    M = 1000
    N = 1000
    targetX = targetSize[1]
    targetY = targetSize[0]
    PadX = int(np.floor(0.5*(N-targetX)))
    PadY = int(np.floor(0.5*(M-targetY)))
    plt.figure(num=2, facecolor='white')
    ax2 = plt.gca()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.imshow(target, cmap='Greys')
    plt.draw()
    plt.pause(0.01)
    resizedTarget = np.lib.pad(Reference[PadY:PadY+targetY, PadX:PadX+targetX], ((PadY, PadY),(PadX,PadX)), 'constant', constant_values=(0,0) )
    target = resizedTarget

    plt.imshow(target, cmap='Greys')
    plt.draw()
    plt.pause(0.01)
    plt.imsave('target.png', target, cmap=plt.cm.gray, vmin=0, vmax = 1)
    return target

#=============================================================================================

def ComputeFilter(target):
    # Compute Filter
    targetFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target)))

    plt.draw()
    plt.pause(0.001)

    # print targetFT
    conjugatePhaseTargetFT = -(np.angle(targetFT))

    binaryConjugatePhaseTargetFT = ((conjugatePhaseTargetFT > 0.0)).astype(int)
    #binaryConjugatePhaseTargetFT = conjugatePhaseTargetFT
    plt.figure(num=3, facecolor='white')
    ax3 = plt.gca()
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.imshow(binaryConjugatePhaseTargetFT, cmap='Greys', interpolation='nearest', vmin=0, vmax=1.0)
  #  plt.savefig('FourierFilter.png', bbox_inches='tight', pad_inches=0)
    plt.draw()
    plt.pause(0.01)
    return binaryConjugatePhaseTargetFT

#=============================================================================================

def CreateAngleImage(rotation):
    # Create a 1000x1000 image of a line (white on black background, and a black on white background)

    LineBW = np.zeros((N,M))
    lines= range(N/2-10, N/2+10)
    LineBW[lines,:] = 1
    LineBW = ndimage.interpolation.rotate(LineBW, rotation, mode='constant', cval=0, reshape=False )
    plt.figure(num=4, facecolor='white')
    plt.imshow(LineBW, cmap='Greys', vmin=0, vmax=1)
    plt.draw()
    plt.imsave('AlignmentBW_'+str(rotation)+'.png', LineBW, cmap=plt.cm.gray, vmin=0, vmax=1)
    LineWB = np.ones((N,M))
    LineWB[lines,:] = 0
    LineWB = ndimage.interpolation.rotate(LineWB, rotation, mode='constant', cval=1, reshape=False)
    plt.figure(num=5, facecolor='white')
    plt.imshow(LineWB, cmap='Greys', vmin=0, vmax=1)
    plt.draw()
    plt.imsave('AlignmentWB_'+str(rotation)+'.png', LineWB, cmap=plt.cm.gray, vmin=0, vmax=1)

#=============================================================================================

def FlipFilter(binaryConjugatePhaseTargetFT):
    return np.flipud(binaryConjugatePhaseTargetFT)
#=============================================================================================


def RotateFilter(binaryConjugatePhaseTargetFT, TranslatedBinaryConjugatePhaseTargetFT, rotation):
    #binaryConjugatePhaseTargetFT is the original filter
    #TranslatedBinaryConjugatePhaseTargetFT is the 180 degree rotated or the flipped version of binaryConjugatePhaseTargetFT
    # depending on what's been passed.
    rotatedBinaryConjugatePhaseTargetFT = ndimage.interpolation.rotate(binaryConjugatePhaseTargetFT, rotation,  mode='constant', cval=0, reshape=True)
    rotated_180_BinaryConjugatePhaseTargetFT = ndimage.interpolation.rotate(TranslatedBinaryConjugatePhaseTargetFT, rotation,  mode='constant', cval=0, reshape=True)
    plt.figure(num=6, facecolor='white')
    plt.subplot(121)
    plt.imshow(rotatedBinaryConjugatePhaseTargetFT, cmap='Greys', vmin=0, vmax=1)
    plt.subplot(122)
    plt.imshow(rotated_180_BinaryConjugatePhaseTargetFT, cmap='Greys', vmin=0, vmax=1)
    plt.draw()
    plt.pause(0.01)
    plt.imsave('FourierFilter_'+str(rotation)+'.png', rotatedBinaryConjugatePhaseTargetFT, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.imsave('FourierFilter_180_'+str(rotation)+'.png', rotated_180_BinaryConjugatePhaseTargetFT, cmap=plt.cm.gray, vmin=0, vmax=1)

#=============================================================================================

def ZeroFilterBlock(binaryConjugatePhaseTargetFT):

    midpointX, midpointY = (0.5*binaryConjugatePhaseTargetFT.shape[0], binaryConjugatePhaseTargetFT.shape[1])

    binaryConjugatePhaseTargetFT[midpointX-0.5*block_size:midpointX+0.5*block_size, midpointY+0.5*block_size:midpointY-0.5*block_size] = 0
    return binaryConjugatePhaseTargetFT
#=============================================================================================

# Define size of filter image (This is determined by the optics)
N = 1000
M = 1000

# Define Zero Block size if used. Set to 0 if no block required
block_size = 0
if len(sys.argv) > 1:
    filterLevel = sys.argv[1]

if len(sys.argv) > 2:
    Angles = []
    if int(sys.argv[2]) == -1:
        Angles = range(0, 91, 5)

    else:
        Angles.append(int(sys.argv[2]))

else:
    Angles = range(0,91,5)


if len(sys.argv) > 3:

    refFilename = sys.argv[3]
    print (refFilename)
    RefPNG, Reference = OpenPNG(refFilename)
    targetFilename = sys.argv[4]
    print (targetFilename)
    targetPNG, target = OpenPNG(targetFilename)
    targetSize = [len(target[0]), len(target)]
    print ('Target Size ', targetSize)
    print ('Reference Size ', len(Reference[0]), len(Reference))
    PadSizeY = int(np.floor(0.5*(M-len(target))))
    PadSizeX = int(np.floor(0.5*(N-len(target[0]))))
    print ('Pad Size ', PadSizeX, PadSizeY)
   # target = np.pad(target, ((PadSizeY, PadSizeY), (PadSizeX, PadSizeX)), mode='constant', constant_values=0)
    plt.figure(0, facecolor='white', figsize=(12,7))
    plt.subplot(121)
    ax0 = plt.gca()
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    plt.imshow(RefPNG) #, interpolation='nearest')
    plt.draw()
    plt.pause(0.01)
    plt.subplot(122)
    ax0 = plt.gca()
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    plt.imshow(targetPNG) #, interpolation='nearest')
    plt.draw()
    plt.pause(0.01)


else:

    EncodeDepth = 12
    # Size of input
    width =  1920
    Rows =   1080
    Reference = np.zeros((Rows, width))
    Reference = np.zeros((Rows, width))

    Field = CF.GenerateSearchField(int( (Rows/EncodeDepth)*width) )
  #  print 'Field ', Field
    Reference = EncodeField(Field)
    ShowMatrix(Reference)

    print ('Reference Size ', Reference.shape)
    # Select a randomly located rectangle of 100, 32
    targetStartRow = randint(0, len(Reference)-100)
    targetEndRow = targetStartRow + 16
    targetStartCol  = randint(0, len(Reference[0])-800)
    targetEndCol    = targetStartCol + 200
    print ('Target Row Start, End ', targetStartRow, targetEndRow)
    print ('Target Col Start, End ', targetStartCol, targetEndCol)
    target = Reference[targetStartRow:targetEndRow, targetStartCol: targetEndCol]
    targetSize = target.shape
    #target = TestField(SequenceLength)
    target = EmbedTarget(Reference, targetSize, target)



# take target out of reference image and pad to target size
binaryConjugatePhaseTargetFT = ComputeFilter(target)
if block_size > 0:
    binaryConjugatePhaseTargetFT = ZeroFilterBlock(binaryConjugatePhaseTargetFT)
    plt.figure(3)
    plt.imshow(binaryConjugatePhaseTargetFT, cmap='Greys', interpolation='nearest', vmin=0, vmax=1.0)
# Flip filter
#flippedBinaryConjugatePhaseTargetFT =  FlipFilter(binaryConjugatePhaseTargetFT)
# Rotate Filter by 180 degrees
#rotatedBinaryConjugatePhaseTargetFT =  ndimage.interpolation.rotate(binaryConjugatePhaseTargetFT, 180,  mode='constant', cval=1, reshape=True)

plt.imsave('FourierFilter.png', binaryConjugatePhaseTargetFT, cmap=plt.cm.gray, vmin=0, vmax=1)

#for i, rotation in enumerate(Angles):
#    RotateFilter(binaryConjugatePhaseTargetFT, rotatedBinaryConjugatePhaseTargetFT, rotation)
#    RotateFilter(binaryConjugatePhaseTargetFT, rotatedBinaryConjugatePhaseTargetFT, -rotation)
#    CreateAngleImage(rotation)
#    CreateAngleImage(-rotation)

print ('Done')


plt.show()