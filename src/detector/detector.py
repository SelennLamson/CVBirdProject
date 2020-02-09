import cv2
import matplotlib.pyplot as plt
import time

from .preprocessing import *
from .corners import *
from .quads import *
from .markers import *


class DetectorParameters:
	def __init__(self):
		# Preprocessing & corners
		self.max_dim = None 				# Max dimension, the other will be rescale
		self.apply_filter = None 			# None, bilinear, gaussian
		self.binary_mode = binary_average   # Binarization function (see preprocessing.py)
		self.binary_mask = None	 			# Function to use to generate a mask for corner detection
		self.corners_scale = 10				# Window in which corners are searched (in pixels)
		self.max_corners = 500				# Maximum number of corner to detect
		self.corners_quality = 0.01			# Minimum treshold for a valid corner

		# Edges
		self.edge_samples = 20				# Number of samples taken along edges to validate them
		self.edge_precision = 0.95			# Percentage of samples that must agree to validate an edge
		self.edge_min_dist = 0.01			# Minimum length of an edge (in percentage of frame's biggest dimension)
		self.edge_max_dist = 0.8			# Maximum length of an edge (in percentage of frame's biggest dimension)
		self.orthogonal_dist = 0.05			# Distance that we walk away from the edge to take samples on both sides (in percentage of edge length)

		# Markers
		self.n_bits = 4						# Number of bits on the side of the markers' content
		self.border = 2						# Width of the markers' border, in bits
		self.aruco_dict = cv2.aruco.DICT_4X4_250	# Aruco dictionnary to use (should be consistent with n_bits)
		self.error_border = 0.1				# Percentage of error allowed on the border
		self.error_content = 0.05			# Percentage of error allowed on the content

		# Debug
		self.draw_preprocessed = False		# Should the detector draw the preprocessed image as debug
		self.draw_binary = False			# Should the detector draw the binary image as debug
		self.draw_mask = False				# Should the detector draw the binary mask as debug
		self.draw_corners = False			# Should the detector overlay detected edges on image
		self.draw_quads = False				# Should the detector overlay detected quads on image
		self.return_preview = False			# Should the detector return the debug images, instead of plotting them directly


def detect_markers(src_img, params: DetectorParameters):
	"""
	Locate markers on an RGB image and optionnaly displays an image with markers' perimeter highlighted and detected corners
	:param src_img:
	:param params:
	:return:
	"""

	start_time = time.time()

	resized = resize(src_img, params.max_dim)
	grayscale, binary, mask = preprocess(resized, apply_filter=params.apply_filter, binary_mode=params.binary_mode, binary_mask=params.binary_mask)
	corners = identify_corners(grayscale, binary, params.corners_scale, params.max_corners, params.corners_quality, mask=mask)
	quads = detect_quads(binary, corners, samples=params.edge_samples, precision=params.edge_precision, min_dist=params.edge_min_dist, max_dist=params.edge_max_dist, orth_dst=params.orthogonal_dist)
	bin_mats = extract_binary_matrices(binary, corners, quads, n_bits=params.n_bits + params.border * 2)
	indexes = binary_check(bin_mats, cv2.aruco.Dictionary_get(params.aruco_dict), params.n_bits, params.border, params.error_border, params.error_content)

	elapsed = time.time() - start_time

	if not (params.draw_preprocessed or
			params.draw_binary or
			params.draw_mask or
			params.draw_corners or
			params.draw_quads or
			params.return_preview):
		return bin_mats, elapsed

	if params.draw_mask and mask is not None:
		preview = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB)
	elif params.draw_binary:
		preview = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2RGB)
	elif params.draw_preprocessed:
		preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
	else:
		preview = resized.copy()

	if params.draw_corners:
		preview = draw_corners(preview, corners)

	if params.draw_quads:
		for c1, c2, c3, c4 in quads:
			cv2.line(preview, tuple(corners[c1]), tuple(corners[c2]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c2]), tuple(corners[c3]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c3]), tuple(corners[c4]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c4]), tuple(corners[c1]), (255, 0, 0), 2)

	if params.return_preview:
		return bin_mats, elapsed, preview

	plt.figure(figsize=(10, 7))
	plt.imshow(preview)
	plt.show()

	return bin_mats, elapsed