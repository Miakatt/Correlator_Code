#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d


def FilterImage(fname):
	Raw = plt.imread(fname)
	plt.figure(1)
	plt.imshow(Raw, cmap='gist_gray', vmin=0, vmax = 10)
	plt.draw()
	plt.pause(0.1)
	# Create a square for the convolution


	ConvImage = convolve2d(Raw, Square)
	# Re-scale image. Divide by area of 'Square'
	ConvArea = np.size(Square)
	print (ConvArea)
	ConvImage /= ConvArea
	plt.figure(2)
	plt.imshow(ConvImage, cmap='gist_gray')
	plt.draw()
	plt.pause(0.1)
	
	return Raw, ConvImage

# Rebin the image after convolution
def rebin(Image, Square):
	n = 0
	average_sum = 0
	new_image = np.zeros([np.shape(Image)[0], np.shape(Image)[1]])
	ImageShape = int(np.shape(Image)[0])

	Skip = int(np.shape(Square)[0])

	for i in np.arange(0, ImageShape,     Skip):
		for k in np.arange(0, ImageShape, Skip):
			average = sum(Image[i:i+Skip, k:k+Skip])/Skip**2
			new_image[i:i+Skip, k:k+Skip] = average
	return new_image



# Filter out pixels that are greater then threshold
def FilterHighs(ConvImage):
	Copy = np.asarray(ConvImage)
	HighPos = np.where(Copy>threshold)
	Copy[HighPos[0],HighPos[1]] = Copy[HighPos[0], HighPos[1]-10]

#	HighFlags = Copy > threshold
#	Copy[HighFlags] = (np.mean(ConvImage[750:1100, 260:800]))	
#	print (np.mean(ConvImage[750:1100, 260:800]))
	return Copy[180:870 , 585:1270]





def PlotCopy(Image):
	plt.figure(3)
	plt.imshow(Image, cmap = 'gist_gray')
	#plt.title('Nothing on Input or Filter')
#	plt.savefig('Background - '+Percentage+'% Saturation.png')
	#plt.savefig('No_Input_No_Filter_Oh_No_Input_No_Filter.png')

	plt.show()


threshold = 7
fname = sys.argv[1]
Percentage = fname[14:16]
Square = np.ones([5,5])
print(fname, Percentage)
Raw, ConvImage = FilterImage(fname)
Copy = FilterHighs(ConvImage)
print (Copy[1,1])
#RebinnedImage = rebin(Copy, Square)
#XYZ = GetXYZ(Copy)
#print(XYZ)

PlotCopy(Copy)

