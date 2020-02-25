import cv2
import numpy as np
import skimage.measure

def extract_binary_matrices(image, corners, quads, n_bits=8):
	"""
	Reads the binary values from detected markers, assuming they are defined anti-clockwise
	:param image: binary image
	:param corners: np(n_corners, 2), coordinates of detected corners
	:param quads: np(n_quads, 4), corners' ids that form quads
	:param n_bits: number of bits on the side of the marker
	:return: bin_mats, np(n_quads, n_bits, n_bits), binary matrix of markers' content
	"""

	nq = quads.shape[0]

	if nq == 0:
		return np.zeros((0, n_bits, n_bits), dtype=bool)

	quad_corners = corners[quads.ravel(), :].reshape(nq, 4, 2)

	maximum_c = np.max(quad_corners, axis=1)
	minimum_c = np.min(quad_corners, axis=1)

	c = np.max(maximum_c - minimum_c, axis=1)
	c -= c % n_bits

	bin_mats = np.zeros((nq, n_bits, n_bits), dtype=bool)
	# compute the perspective transform matrix and then apply it
	for i in range(nq):
		dst = np.array([[0, 0], [c[i] - 1, 0], [c[i] - 1, c[i] - 1], [0, c[i] - 1]])
		M = cv2.getPerspectiveTransform(quad_corners[i].astype(np.float32), dst.astype(np.float32))
		warped = cv2.warpPerspective(image, M, (c[i], c[i]))

		# We apply a average pooling on every sub-square of the marker (hopefully bits)
		c_bit = c[i] // n_bits
		bin_mats[i] = skimage.measure.block_reduce(warped, (c_bit, c_bit), lambda x, axis: np.sum(x, axis=axis) / c_bit ** 2 > 0.5).astype(bool)
	return bin_mats


def binary_check(bin_mats, aruco_dict, c_bits, border, error_border, error_content):
	s = 2*border + c_bits									# Total size of the marker side
	nm = bin_mats.shape[0]									# Number of markers
	nb = (c_bits**2)//8 + (1 if (c_bits**2) % 8 > 0 else 0)	# Number of bytes to represent data

	if nm == 0:
		return [], []

	byteslist = aruco_dict.bytesList.ravel()
	dic_bits = np.unpackbits(byteslist).reshape(len(aruco_dict.bytesList), 4, nb*8)

	borders = bin_mats.copy()
	borders[:, border:border+c_bits, border:border+c_bits] = 0
	borders = np.sum(borders.reshape((nm, -1)), axis=1)
	borders = borders < (s**2 - c_bits**2) * error_border

	content = bin_mats[:, border:border+c_bits, border:border+c_bits].reshape((nm, c_bits**2))
	flatten = np.zeros((nm, nb*8), dtype=int)
	flatten[:, nb*8-c_bits**2:] = content

	errors = np.sum(flatten[:, np.newaxis, np.newaxis, :] != dic_bits[np.newaxis, :, :, :], axis=3)
	best_orientations = np.argmin(errors, axis=2)
	errors = np.min(errors, axis=2)

	index = np.argmin(errors, axis=1)
	best_orientations = np.array([best_orientations[i, index[i]] for i in range(len(index))], dtype=int)
	best_orientations = (4 - best_orientations) % 4

	errors = np.min(errors, axis=1)
	passes = np.bitwise_and(errors / c_bits**2 < error_content, borders)

	return passes * index - 1 + passes, best_orientations

