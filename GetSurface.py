import numpy as np 
import matplotlib.pyplot as plt 
import csv
import sys
import scipy.linalg
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def OpenFile(fname):
	with open(fname, 'r') as f:
		next(f)
		reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
		for row in reader:
			X_coord.append(row[0])
			Y_coord.append(row[1])
			Z_coord.append(row[2])

	return X_coord, Y_coord, Z_coord	

#======================================================
def func2(C, XX, YY):
	# Try to fit a 2nd order polynomial to X and Y
	XF = XX.flatten()
	YF = YY.flatten()

	return np.dot(np.c_[np.ones((XF).shape), XF, YF, XF*YF, XF**2, YF**2], C).reshape(XX.shape)

#======================================================
# def func3(C, XX, YY):
# 	# Try to fit a 4th order polynomial to X and Y
# 	XF = XX.flatten()
# 	YF = YY.flatten()

# 	return np.dot(np.c_[np.ones((XF).shape), XF, YF, XF*YF, \
# 		XF**2, YF**2,       \
# 		XF**3, YF**3], C).reshape(XX.shape)


#======================================================
def FitData(X,Y,Z, XX, YY):
	zi = griddata(X, Y, Z, XX, YY, interp='linear')	
	fig3 = plt.figure(3)
	ax3 = plt.gca(projection='3d')
	CS = ax3.plot_surface(XX, YY, zi, cmap='inferno', rstride = 1, cstride = 1, vmin=np.min(ZZ), vmax=np.max(ZZ))
	ax3.scatter(X, Y, Z, cmap = 'inferno')
	ax3.set_zlim3d(np.min(ZZ)*0.8, np.max(ZZ)*1.2)
	plt.title('Triangulation Fit')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Triangulation_Fit.png')
	plt.draw()
	plt.pause(0.01)
#======================================================

def GetSurface(X, Y, Z):
	Ext = 50
	data = np.c_[X, Y, Z]
	mn = np.min(data, axis=0)
	mx = np.max(data, axis=0)
	XX,YY = np.meshgrid(np.linspace(mn[0]-Ext, mx[0]+Ext, 20), np.linspace(mn[1]-Ext, mx[1]+Ext, 20))
	
	A = np.c_[np.ones(data.shape[0]), data[:,:2] , np.prod(data[:,:2], axis=1) , data[:,:2]**2]
	C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
	print (C)
	# Order-2 polynomial
	ZZ = func2(C, XX, YY)
	
	fig2 = plt.figure(2)
	ax = fig2.gca(projection='3d')
	ax.plot_surface(XX,YY,ZZ, cmap='inferno', rstride = 1, cstride = 1, alpha = 0.6 )#, vmin=40, vmax = 200)
	ax.set_zlim3d(np.min(ZZ)*0.8, np.max(ZZ)*1.2)
	plt.title('2nd Order Polynomial Fit.  %s%% saturation ' % Percentage )
	plt.xlabel('x')
	plt.ylabel('y')

	ax.scatter(X, Y, Z, cmap = 'inferno')
	plt.draw()
	plt.savefig('2-order_polyfit.png') # '+Percentage+'pc.png')

	
# #--------------------------
# 	A = np.c_[np.ones(data.shape[0]), data[:,:2] , np.prod(data[:,:2], axis=1) , data[:,:2]**3]
# 	C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
# 	print (C)
# 	C = np.append(C, [0,0])
# 	print (C)
# 	ZZ = func3(C, XX, YY)
# 	fig3 = plt.figure(3)
# 	ax3 = fig3.gca(projection='3d')
# 	ax3.plot_surface(XX, YY, ZZ , cmap='inferno', rstride=1, cstride=1, alpha = 0.6)
# 	ax3.scatter(X,Y,Z, cmap='inferno')
# 	plt.title('Least Squared Fit, 3rd Order Polynomial')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.draw()
# 	plt.savefig('4-order_polyfit.png')

	return XX, YY, ZZ

#======================================================

X_coord = []
Y_coord = []
Z_coord = []

fname = sys.argv[1]	
Percentage = (fname[5:7])
X, Y, Z = OpenFile(fname)

XX, YY, ZZ = GetSurface(X,Y,Z)
#FitData(X,Y,Z,XX,YY)
plt.show()