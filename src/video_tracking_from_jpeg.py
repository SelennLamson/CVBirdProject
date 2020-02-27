from tqdm import tqdm
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from detector.detector import detect_markers, DetectorParameters
from detector.preprocessing import *
from CameraPosTracker import CameraPosTracker
from model.Frame import Frame


base_path = '../data/simulation/videos/bottom_easier_600x600'
treated_path = '../data/treated_videos'
frame_pattern = 'frame_bottom_'
frames = os.listdir(base_path)
nb_frames = len(frames)

print("Processing {} frames".format(nb_frames))

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

# initiate the camera tracker
ref_frame = Frame(0)
ref_frame.addMarkersAndRotFromJson(view='bottom', filterVisible=False, addMarkers=True)
camera_pos_tracker = CameraPosTracker(ref_frame)

# Launching of tracking
start_frame = int(nb_frames * start_percent // 1)
end_frame = int(nb_frames * end_percent // 1)
range_it = tqdm(range(start_frame, end_frame))

for i in range_it:
	frame = cv2.imread(f'{base_path}/{frame_pattern}{i}.jpg')
	if i < start_frame:
		continue
	if i > end_frame:
		break

	if not (frame is None):
		markers, elapsed, out_frame = detect_markers(frame, params)
		print('frame', i, '- len markers', len(markers))

		# compute camera location from the markers of the current frame
		curr_frame = Frame(i)
		curr_frame.addMarkersAndRotFromJson(view='bottom')
		curr_frame.markers = markers

		try:
			camera_pos_tracker.computeMoveFromRefFrame(curr_frame, debug=False)
		except ValueError:
			plt.imshow(out_frame)
			plt.show()
			print([m.corners for m in markers])
			print([m.id for m in markers])


		if out is None:
			out = cv2.VideoWriter(treated_path + '/treated_' + base_path.split('/')[-1] + '.avi',
								  apiPreference=cv2.CAP_FFMPEG,
								  fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
								  fps=1,
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

camera_pos_tracker.plotTrajectory()
