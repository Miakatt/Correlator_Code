#!/usr/python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from PIL import Image

def OpenFile(arg):
    fullfilename = sys.argv[arg]
    filename, file_ext = os.path.splitext(sys.argv[arg])
    print(filename, file_ext)
    return fullfilename, file_ext


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def WrapArray(ChemArray, TargetShape):

    if len(ChemArray) != 1024:
        print ('Chemical Array Signature has unexpected length. Should be 1024.')
        sys.exit()

    else:
        ShapedChemArray = np.reshape(ChemArray, (TargetShape))
        return ShapedChemArray

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def Resize(ShapedChemArray, n):
	# Currently unused. Scales the original data array by n
    pixel = np.zeros((n, n))
    pixel[0,:] = 1
    pixel[:,2] = 1
    #plt.imshow(pixel)
    #plt.show()

    ResizedChemArray = np.kron(ShapedChemArray, pixel )

    ShapedChemArray = ResizedChemArray
    return ShapedChemArray

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def ManipulateArray(ShapedChemArray):
	# Changes 0/1 binary shaped chemical array. Replaces zeros with a right-angle symbol
	# and ones with a chequerboard symbol.
	ManipulatedArray = np.zeros((n*ShapedChemArray.shape[0], n*ShapedChemArray.shape[1]))
	ZeroReplace = [[1,1],[0,1]] #,[0,0,1]]
	OneReplace  = [[1,0],[1,1]] # ,[0,1,0]]
	for it in range(1024):
		coords = np.unravel_index(it, (TargetShape))
		if ShapedChemArray[coords] == 0:
			ManipulatedArray[n*coords[0]:n*coords[0]+n,n*coords[1]:n*coords[1]+n] = ZeroReplace
		elif ShapedChemArray[coords] == 1:	
			ManipulatedArray[n*coords[0]:n*coords[0]+n,n*coords[1]:n*coords[1]+n] = OneReplace
	return ManipulatedArray		


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def DisplayChemArray(ShapedChemArray, ChemID):
    plt.figure(1)
    plt.cla()
    plt.imshow(ShapedChemArray, cmap='gist_gray', vmin=0, vmax=1)
   # plt.pause(0.1)
    SaveSingleImage(ShapedChemArray ,  ChemID)

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def SaveSingleImage(ThisArray, ThisChemID):
	ImageArray = ThisArray
	I8 = (((ImageArray - ImageArray.min()) / ImageArray.max() - ImageArray.min()) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img.save(ImagePath+ThisChemID+'.png', vmin=0, vmax=1)
#    plt.imsave( ImagePath+ThisChemID+'.png', ThisArray.astype(np.uint8), cmap='gist_gray', vmin=0, vmax=1)

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
	#  plt.imsave( ImagePath+'Reference_Page_'+str(N).zfill(4)+'.png', Ref.astype(np.uint8), cmap='gist_gray', vmin=0, vmax=1)
	ImageArray = Ref
	I8 = (((ImageArray - ImageArray.min()) / ImageArray.max() - ImageArray.min()) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img.save(ImagePath+'Reference_Page_'+str(N).zfill(4)+'.png', vmin=0, vmax=1)


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def GenerateRandomBackground(Shape, Saturation):
	RandomBackground = np.zeros(Shape)
	RandomBackground = np.random.choice([0,1], size=Shape, p=[Saturation, 1.0-Saturation])
	return RandomBackground


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo


def MakeReferencePage(Ref,  ShapedChemArray, j, k):

    XStepStart = k * (ShapedChemArray.shape[1] )
    XStepEnd = XStepStart + ShapedChemArray.shape[1]
    YStepStart = j * (ShapedChemArray.shape[0] )
    YStepEnd = YStepStart + ShapedChemArray.shape[0]
    Ref[XStepStart:XStepEnd, YStepStart:YStepEnd] = ShapedChemArray
    plt.figure(2)
    plt.cla()
    ax1 = plt.gca()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.imshow(Ref, cmap='gist_gray', vmin=0, vmax=1)
    #plt.draw()
    #plt.pause(0.1)



# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def InitializeReference(RefShape):
    Ref = np.zeros(RefShape)
    return Ref

#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def EmbedTarget(ShapedChemArray, PadX, PadY):

    plt.figure(num=1, facecolor='white')
    ax2 = plt.gca()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    PaddedChemArray = np.pad(ShapedChemArray, ((PadY, PadY), (PadX,PadX)), 'constant', constant_values=(0,0) )

    return PaddedChemArray

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def MergeArrays(ManipulatedArray, RandomBackground):
	MergedArrays = (ManipulatedArray != RandomBackground).astype(int)
	return MergedArrays

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def ComputeAndSaveFilter(PaddedChemArray, ChemID):
	ChemFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(PaddedChemArray)))
	ConjFT = -(np.angle(ChemFT))
	binaryConjFT =(ConjFT > 0.1).astype(np.uint8)
	if (ApplyBlock):
	    binaryConjFT = DCBlock(binaryConjFT)
	ax3 = plt.gca()
	ax3.xaxis.set_visible(False)
	ax3.yaxis.set_visible(False)
	ax3.set_xticks([])
	ax3.set_yticks([])
	plt.imshow(binaryConjFT)
	#    plt.imsave(ImagePath+'Filter_'+str(ChemID)+'.png', binaryConjFT, cmap='gist_gray', vmin=0, vmax=1)

	ImageArray = binaryConjFT
	I8 = (((ImageArray - ImageArray.min()) / ImageArray.max() - ImageArray.min()) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img.save(ImagePath+'Filter_'+str(ChemID)+'.png', vmin=0, vmax=1)


#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

def DCBlock(filter):
    center = (0.5*filter.shape[0], 0.5*filter.shape[1])
    Block = np.zeros([DC_BlockSize, DC_BlockSize])
    x1 = int(center[0]-0.5*DC_BlockSize)
    x2 = int(center[1]+0.5*DC_BlockSize)
    filter[x1:x2, x1:x2] = Block
    return filter

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
def MakeReferenceCSV(csvFile, ChemRow):

	myCSV = open(csvFile, 'a')
	with myCSV:
	#	print ('Writing Row of Chem CSV')
		#print (ChemRow)
		writer.writerow(ChemRow)
		ChemRow = []

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo



j = 0
k = 0
PageNo = 0
w =1

#ImagePath = '/mnt/cloud_data/Cheminformatics_Images/'
ImagePath = './SingleImages/'
csvFile = 'Chem_Positions_'+str(PageNo)+'.csv'
ChemRow = []
print ('Opening CSV file...')
myCSV = open(csvFile, 'w')
writer = csv.writer(myCSV)
		

if not os.path.exists(ImagePath):
    os.makedirs(ImagePath)
RegenerateRandom = input('Would you like to regenerate a new random background? [Y/N] ')
if RegenerateRandom=='Y' or RegenerateRandom=='y':
	GenRandom = True
else:
	GenRandom = False

Saturation = 0.95
ApplyBlock = True
DC_BlockSize = 10
n = 2
TargetShape = [32,32]       # Ensure this wraps a 1024bit image correctly.
RefShape    = [1024, 1280]

targetX = TargetShape[1]
targetY = TargetShape[0]
PadX = int(np.floor(0.5 * (1000 - n*targetX)))
PadY = int(np.floor(0.5 * (1000 - n*targetY)))


Ref = np.zeros(RefShape)
fullfilename, ext = OpenFile(1)
print('Reading CHEM File - ', fullfilename)


# Generate random 2D array to OR with the shaped chemical array
# This is to reduce the periodicity and get truer peaks according to Alex, who knows this stuff
# 'cos he studied it at a uni with oak wall panels, so you know it's posh! 
if (GenRandom):
	RandomBackground = GenerateRandomBackground([n*TargetShape[0], n*TargetShape[1]], Saturation)
	plt.imshow(RandomBackground, cmap='gist_gray_r', vmin=0, vmax=1)
	plt.draw()
	plt.pause(1)
	plt.imsave(ImagePath+'RandomBackground.png', RandomBackground, vmin=0, vmax=1)
	np.save(ImagePath+'RandomBackground', RandomBackground)
else:
	RandomBackground = np.load(ImagePath+'RandomBackground.npy')
	plt.imshow(RandomBackground, cmap='gist_gray_r')
	plt.draw()
	plt.pause(1)
	plt.gcf()
	plt.close('all')
	print (RandomBackground[1,1])
MergedArrays = np.zeros((n*TargetShape[0], n*TargetShape[1]))
with open(fullfilename,'r') as f:

    for line in f:
        w += 1
        ChemID = line.split("\t")[0]
        ChemString = line.split("\t")[1].rstrip()
        ChemArray = (list(map(int, ChemString))) # Convert string of numbers to list of ints.
        

        ShapedChemArray = WrapArray(ChemArray, TargetShape)
        SaveSingleImage(ShapedChemArray, ChemID)
     #   ShapedChemArray = Resize(ShapedChemArray, n)
        ManipulatedArray = ManipulateArray(ShapedChemArray)
        MergedArrays = MergeArrays(ManipulatedArray, RandomBackground)
        #print (MergedArrays.shape)
      #  DisplayChemArray(MergedArrays, ChemID)
        SaveSingleImage(MergedArrays, ChemID+'Encoded')

        PaddedChemArray = EmbedTarget(MergedArrays, PadX, PadY)
        ComputeAndSaveFilter(PaddedChemArray, ChemID)

      #  print (ChemID)

        MakeReferencePage(Ref, MergedArrays, j, k)

        j += 1
        ChemRow.append(str(ChemID))
        if j == RefShape[1]/(n*TargetShape[1]):
        	MakeReferenceCSV(csvFile, ChemRow)
        	ChemRow = []
        	j = 0
        	k += 1
        if (k >= RefShape[0]/(MergedArrays.shape[0])) : #and (j >= RefShape[1]/(ManipulatedArray.shape[1])-1):
            SaveReferenceImage(Ref, PageNo)
            j = 0
            k = 0
            PageNo +=1
            Ref = np.zeros(RefShape)
            myCSV.close()
            # Start new csv file
            csvFile = 'Chem_Positions_'+str(PageNo)+'.csv'
            ChemRow = []
            print ('Opening CSV file...')
            myCSV = open(csvFile, 'w')
            writer = csv.writer(myCSV)

    #    if w > 10:

    SaveReferenceImage(Ref, PageNo)
#    writer.close()
    #        sys.exit()
