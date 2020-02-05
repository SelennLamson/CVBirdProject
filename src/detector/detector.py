import cv2
import matplotlib.pyplot as plt
import time

from .preprocessing import *
from .corners import *
from .quads import *
from .markers import *


def detect_markers(src_img, max_dim=None, draw_preprocessed=False, draw_corner=False, draw_quads=False):
	"""
	Locate markers on an RGB image and optionnaly displays an image with markers' perimeter highlighted and detected corners
	:param src_img:
	:param max_dim:
	:param draw_preprocessed:
	:param draw_corner:
	:param draw_quads:
	:return:
	"""

	start_time = time.time()

	resized = resize(src_img, max_dim)
	grayscale, binary = preprocess(resized, apply_bilateral=True, binary_mode=binary_average)
	corners = identify_corners(grayscale, binary, scale=10, mask=None)
	quads = detect_quads(binary, corners, samples=20, precision=0.95, min_dist=0.01, max_dist=0.8)
	bin_mats = extract_binary_matrices(binary, corners, quads, n_bits=8)
	indexes = binary_check(bin_mats, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 4, 2, 0.1, 0.05)
	print(indexes)

	elapsed = time.time() - start_time

	if not (draw_preprocessed or draw_corner or draw_quads):
		return bin_mats

	if draw_preprocessed:
		preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
	else:
		preview = resized.copy()

	if draw_corner:
		preview = draw_corners(preview, corners)

	if draw_quads:
		for c1, c2, c3, c4 in quads:
			cv2.line(preview, tuple(corners[c1]), tuple(corners[c2]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c2]), tuple(corners[c3]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c3]), tuple(corners[c4]), (255, 0, 0), 2)
			cv2.line(preview, tuple(corners[c4]), tuple(corners[c1]), (255, 0, 0), 2)

	plt.figure(figsize=(10, 7))
	plt.imshow(preview)
	plt.show()

	return bin_mats, elapsed
