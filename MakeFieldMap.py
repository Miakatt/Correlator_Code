#!/usr/python3

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from scipy.misc import imread
import random
import shutil
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def OpenImageFile(png):
	im = imread(png, flatten=True)

	return im

#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def GenerateRandomData(rawDataShape):
	# Generate random data with varied saturation. 
	rawData = np.zeros(rawDataShape)
	rawData = np.random.choice([0,1], size=rawDataShape, p=[0.1,0.9])
	return rawData

#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def GenerateRandomBackground(Shape, Saturation):
	RandomBackground = np.zeros(Shape)
	RandomBackground = np.random.choice([0,1], size=Shape, p=[Saturation, 1.0-Saturation])
	return RandomBackground

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def WrapArray(rawData, Shape):

    if np.size(rawData) != 1024:
        print ('Chemical Array Signature has unexpected length. Should be 1024.')
        sys.exit()

    else:
        ShapedChemArray = np.reshape(rawData, (Shape))
        return ShapedChemArray


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def ManipulateArray(shapedRawData, Shape):
	# Changes 0/1 binary shaped chemical array. Replaces zeros with a right-angle symbol
	# and ones with a chequerboard symbol.
	ManipulatedArray = np.zeros((n*shapedRawData.shape[0], n*shapedRawData.shape[1]))
	ZeroReplace = [[1,1],[0,1]] #,[0,0,1]]
	OneReplace  = [[1,0],[1,1]] # ,[0,1,0]]
	for it in range(1024):
		coords = np.unravel_index(it, (Shape))
		if shapedRawData[coords] == 0:
			ManipulatedArray[n*coords[0]:n*coords[0]+n,n*coords[1]:n*coords[1]+n] = ZeroReplace
		elif shapedRawData[coords] == 1:	
			ManipulatedArray[n*coords[0]:n*coords[0]+n,n*coords[1]:n*coords[1]+n] = OneReplace
	return ManipulatedArray		

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def MakeReferencePage(Ref,  ImageInput, j, k):

	YStepStart = k * (ImageInput.shape[1] )
	YStepEnd = YStepStart + ImageInput.shape[1] 

	XStepStart = j * (ImageInput.shape[0] )
	XStepEnd = XStepStart + ImageInput.shape[0]  
	
	Ref[YStepStart:YStepEnd, XStepStart:XStepEnd] = ImageInput
	plt.figure(2)
	plt.cla()
	ax1 = plt.gca()
	ax1.xaxis.set_visible(False)
	ax1.yaxis.set_visible(False)
	ax1.set_xticks([])
	ax1.set_yticks([])

	plt.imshow(Ref, cmap='gist_gray', vmin=0, vmax=1)
	#plt.draw()
	#plt.pause(0.01)
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def EmbedTarget(RandomArray, PadX, PadY):

    plt.figure(num=1, facecolor='white')
    ax2 = plt.gca()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    PaddedArray = np.pad(RandomArray, ((PadY, PadY), (PadX,PadX)), 'constant', constant_values=(0,0) )

    return PaddedArray


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def ComputeAndSaveFilter(PaddedArray, N):
    ChemFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(PaddedArray)))
    ConjFT = -(np.angle(ChemFT))
    binaryConjFT =(ConjFT > 0.1).astype(np.uint8)
   # if (ApplyBlock):
   #     binaryConjFT = DCBlock(binaryConjFT)
    ax3 = plt.gca()
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.imshow(binaryConjFT)
    I  = binaryConjFT
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8)

    img.save(FilterImagePath+'Random_Filter_'+str(N)+'.png')
#    plt.imsave(ImagePath+'Random_Filter_'+str(N)+'.png', binaryConjFT, cmap='gist_gray', vmin=0, vmax=1)

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def SaveReferenceImage(Ref, N):
	plt.figure(2)
	ax1 = plt.gca()
	ax1.cla()
	ax1.xaxis.set_visible(False)
	ax1.yaxis.set_visible(False)
	ax1.set_xticks([])
	ax1.set_yticks([])
	plt.imshow(Ref, cmap='gist_gray', vmin=0, vmax=1)
	I  = Ref
	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img.save(InputImagePath+'Reference_Fieldmap_Random_'+str(N)+'.png')
	#  plt.imsave( ImagePath+'Reference_Fieldmap_Random_'+str(N)+'.png', Ref.astype(np.uint8), cmap='gist_gray', vmin=0, vmax=1)
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def MergeArrays(ManipulatedArray, RandomBackground):
	MergedArrays = (ManipulatedArray != RandomBackground).astype(int)
	return MergedArrays

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def SaveImage(RandomBackground, N):
	I  = RandomBackground
	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img.save(ChemImagePath+'Random_Image_'+str(N)+'.png')

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def MergeArrays(ManipulatedArray, RandomBackground):
	MergedArrays = (ManipulatedArray != RandomBackground).astype(int)
	return MergedArrays

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

# Read random 2D array to OR with the shaped chemical array
# This is to reduce the periodicity and get truer peaks according to Alex, who knows this stuff
# 'cos he studied it at a uni with oak wall panels, so you know it's posh! 
#RandomBackground = np.load('./SingleImages/RandomBackground.npy')
#plt.imshow(RandomBackground, cmap='gist_gray_r')
#plt.draw()
#plt.pause(1)
#plt.gcf()
#plt.close('all')

n = 2
j = 0
k = 0
PageNo = 0
N = 0

rawDataShape = [1,1024]
RefShape = [1024, 1280]
Ref = np.zeros(RefShape)


Shape = (32,32)

PadX = int(np.floor(0.5 * (1000 - n*Shape[1])))
PadY = int(np.floor(0.5 * (1000 - n*Shape[0])))

# Generate N Random, Unique images. If False, Creates N copies of one random image.
GenRandom = True
#FilterImagePath = '/home/optalysys/OptalysysSoftware/externalapi/Samples/ChemInformatics/SampleData/PretransformedFilters/'
#InputImagePath  = '/home/optalysys/OptalysysSoftware/externalapi/Samples/ChemInformatics/SampleData/Inputs/'
#ChemImagePath   = '/home/optalysys/OptalysysSoftware/externalapi/Samples/ChemInformatics/SampleData/ChemImage/'
FilterImagePath =  './'
InputImagePath  =  './'
ChemImagePath   =  './'

try:
    os.remove(FilterImagePath+'*')
except OSError:
    pass

try:
    os.remove(ChemImagePath+'*')
except OSError:
    pass

try:
    os.remove(InputImagePath+'*')
except OSError:
    pass




while N<1:
	CompleteFlag = False
	Saturation = 0.65 # random.uniform(0.2, 0.6)
	if GenRandom:
		RandomBackground = GenerateRandomBackground(Shape, Saturation)
	elif not GenRandom and N == 0:
		RandomBackground = GenerateRandomBackground(Shape, Saturation)


	SaveImage(RandomBackground, N)
	# Create a dummy data array
	rawData = GenerateRandomData(rawDataShape)
	# Wrap it
	shapedRawData = WrapArray(rawData, Shape)

	# Merge dummy data and random data
	dummyData = MergeArrays(shapedRawData, RandomBackground)

	# Encode array
	ManipulatedArray = ManipulateArray(dummyData, Shape)
	print (np.shape(ManipulatedArray))
	PaddedArray = EmbedTarget(ManipulatedArray, PadX, PadY)
	ComputeAndSaveFilter(PaddedArray, N)
	while CompleteFlag is False:
		
		MakeReferencePage(Ref, ManipulatedArray, j, k)
		j += 1
		if (j == RefShape[1]/ManipulatedArray.shape[1]):
			j = 0
			k += 1
			
		if (k >= RefShape[0]/ManipulatedArray.shape[0]): #and (j >= RefShape[1]/(ManipulatedArray.shape[1])-1):
			SaveReferenceImage(Ref, PageNo)
			j = 0
			k = 0
			PageNo +=1
			Ref = np.zeros(RefShape)
			CompleteFlag = True
	N += 1
	print(N)
