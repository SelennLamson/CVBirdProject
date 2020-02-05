import cv2
from cv2 import aruco
import numpy as np

ardict = aruco.Dictionary_get(aruco.DICT_4X4_250)

print(ardict.bytesList.shape)
# N_MARKERS x N_ x 4

byteslist = ardict.bytesList.ravel()#.reshape(250, 4, 2)
arr = np.unpackbits(byteslist).reshape(250, 4, 16)




m = arr[5]

print(m)

for i in m:
	for a in range(4):
		print(i[a*4:a*4+4])
	print("---")

# for i in range(4):
# 	m1 = m[i]
#
# 	for b in m1:
# 		s = bin(b)[2:].rjust(8, '0')
# 		print(' '.join(s[0:4]) + '\n' + ' '.join(s[4:]) + '  ' + str(b))
#
# 	print("---")

# HERE WE HAVE THE FOUR ROTATIONS
# There is an error in opencv python implementation because of differences with the way C stores arrays
# The shape of the dictionary is (eg. 4x4_250):
# - 250, number of markers
# - 2, number of bytes necessary to encode the markers content
# - 4, number of rotations for each marker

# To fix the problem, dict bytes should be taken with:
# byteslist = dic.bytesList.ravel().reshape(250, 4, 2)

# Now, byteslist[142, 3, :] gives the bytes of the third rotation of marker 142
