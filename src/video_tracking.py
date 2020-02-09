from tqdm import tqdm
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from detector.detector import detect_markers, DetectorParameters
from detector.preprocessing import *


base_path = '../data/videos'
treated_path = '../data/treated_videos'
videos = os.listdir(base_path)

while True:
	try:
		print("Videos in " + base_path + ":")
		for i, n in enumerate(videos):
			print("- [{}] ".format(i) + n)
		ans = int(input("Select a video: "))
		assert 0 <= ans < len(videos)
		break
	except (ValueError, AssertionError):
		pass

vidcap = cv2.VideoCapture(base_path + '/' + videos[ans])
nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)

print("Processing {} frames at {:.2f} fps.".format(nb_frames, fps))

out = None

min_elapsed = max_elapsed = None
avg_elapsed = 0

start_percent = 0
end_percent = 1

params: DetectorParameters = DetectorParameters()
params.max_dim = 800
params.draw_binary = False
params.draw_preprocessed = False
params.draw_corners = True
params.draw_quads = True
params.draw_mask = False
params.return_preview = True

params.apply_filter = 'bilateral'
params.binary_mode = binary_midgray
params.binary_mask = intensity_mask
params.corners_scale = 6
params.edge_samples = 20
params.edge_precision = 0.95
params.edge_max_dist = 0.15
params.orthogonal_dist = 0.05


# Launching of tracking

start_frame = int(nb_frames * start_percent // 1)
end_frame = int(nb_frames * end_percent // 1)
range_it = tqdm(range(end_frame - start_frame))

for i in range_it:
	success, frame = vidcap.read()

	if i < start_frame:
		continue
	if i > end_frame:
		break

	if success:
		bin_mats, elapsed, out_frame = detect_markers(frame, params)

		if out is None:
			out = cv2.VideoWriter(treated_path + '/treated_' + videos[ans],
								  apiPreference=cv2.CAP_FFMPEG,
								  fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
								  fps=fps,
								  frameSize=(out_frame.shape[1], out_frame.shape[0]),
								  isColor=True)

		if min_elapsed is None or elapsed < min_elapsed:
			min_elapsed = elapsed
		if max_elapsed is None or elapsed > max_elapsed:
			max_elapsed = elapsed
		avg_elapsed += elapsed

		out.write(np.uint8(out_frame))
	else:
		print('Issue to read frame id:', i)

avg_elapsed /= nb_frames

print(("Processing finished.\n" +
	  "- Min. time on frame: {:.4f}s\n" +
	  "- Max. time on frame: {:.4f}s\n" +
	  "- Average time: {:.4f}s")
	  .format(min_elapsed, max_elapsed, avg_elapsed))

out.release()