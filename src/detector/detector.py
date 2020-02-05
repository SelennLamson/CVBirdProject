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
		self.max_dim = None
		self.apply_bilateral = True
		self.binary_mode = binary_average
		self.binary_mask = None
		self.corners_scale = 10
		self.max_corners = 500
		self.corners_quality = 0.01

		# Edges
		self.edge_samples = 20
		self.edge_precision = 0.95
		self.edge_min_dist = 0.01
		self.edge_max_dist = 0.8
		self.orthogonal_dist = 0.05

		# Markers
		self.n_bits = 4
		self.border = 2
		self.aruco_dict = cv2.aruco.DICT_4X4_250
		self.error_border = 0.1
		self.error_content = 0.05

		# Debug
		self.draw_preprocessed = False
		self.draw_binary = False
		self.draw_mask = False
		self.draw_corners = False
		self.draw_quads = False
		self.return_preview = False


def detect_markers(src_img, params: DetectorParameters):
	"""
	Locate markers on an RGB image and optionnaly displays an image with markers' perimeter highlighted and detected corners
	:param src_img:
	:param params:
	:return:
	"""

	start_time = time.time()

	resized = resize(src_img, params.max_dim)
	grayscale, binary, mask = preprocess(resized, apply_bilateral=params.apply_bilateral, binary_mode=params.binary_mode, binary_mask=params.binary_mask)
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
