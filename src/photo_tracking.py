import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from detector.detector import *


base_path = '../data/simulation/videos/bottom_600x600'
images = os.listdir(base_path)

while True:
	try:
		print("Pictures in " + base_path + ":")
		for i, n in enumerate(images):
			print("- [{}] ".format(i) + n)
		ans = int(input("Select a picture: "))
		assert 0 <= ans < len(images)
		break
	except (ValueError, AssertionError):
		pass

params: DetectorParameters = DetectorParameters()
params.max_dim = 800
params.draw_binary = False
params.draw_preprocessed = False
params.draw_corners = True
params.draw_quads = True
params.draw_mask = False

params.apply_filter = 'bilateral'
params.binary_mode = binary_midgray
params.binary_mask = intensity_mask
params.corners_scale = 6
params.edge_samples = 20
params.edge_precision = 0.95
params.edge_max_dist = 0.15
params.orthogonal_dist = 0.05

src_img = cv2.imread(base_path + '/' + images[ans])
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
markers, elapsed = detect_markers(src_img, params)
print("Elapsed time: %.3fs" % elapsed)

print(markers)

