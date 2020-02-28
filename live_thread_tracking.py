import numpy as np
import threading
import time
from typing import List
import cv2

from src.model.Marker import Marker
from src.detector.detector import detect_markers
from src.detector.detector import DetectorParameters
from src.detector.preprocessing import binary_midgray, intensity_mask


class DetectorThread(threading.Thread):
	def __init__(self, name, w, h, cam_no, receiver):
		super(DetectorThread, self).__init__(name=name)
		self.stop_event = threading.Event()
		self.detect_event = threading.Event()

		self.w = w
		self.h = h
		self.cam_no = cam_no
		self.imlen = h * w
		self.image = np.zeros(self.imlen, dtype=np.uint8)
		self.head = 0

		self.receiver = receiver

		self.params: DetectorParameters = DetectorParameters()
		self.params.max_dim = 800
		self.params.apply_filter = 'bilateral'
		self.params.binary_mode = binary_midgray
		self.params.binary_mask = intensity_mask
		self.params.corners_scale = 6
		self.params.edge_samples = 20
		self.params.edge_precision = 0.95
		self.params.edge_max_dist = 0.15
		self.params.orthogonal_dist = 0.05

	def run(self):
		while not self.stop_event.is_set():
			if self.detect_event.is_set():
				self.detect_markers()
				self.detect_event.clear()
			time.sleep(0.01)

	def stop(self):
		self.stop_event.set()

	def ask_detection():
		self.detect_event.set()

	def feed_image(self, feed):
		flen = len(feed)
		if self.head + flen >= self.imlen:
			self.image[self.head:] = feed[:self.imlen - self.head]
		else:
			self.image[self.head: self.head + flen] = feed

		self.head += flen
		if self.head >= self.imlen:
			self.head = 0

	def detect_markers(self):
		# cv2.imwrite('D:/Thomas/UnrealProjects/UE4.22/MarkerDetection/Source/Python/frame{}.png'.format(int((time.time()*1000)%1000)), self.image.copy().reshape(self.h, self.w))

		# Preprocessing
		# gray = self.image.copy().reshape(self.h, self.w)  # Conversion to gray scale
		img = cv2.imread(
			'D:/Thomas/UnrealProjects/UE4.22/MarkerDetection/Source/Python/frame_' + str(self.cam_no) + '.jpg')

		markers, elapsed = detect_markers(img, self.params)

		absolute_location = [0.0, 0.0, 0.0]

		# Building result string:
		# --> 'Results:X,Y,Z|101,x,y,z,rx,ry,rz|3,x,y,z,rx,ry,rz'
		r = 'Results:'
		r += ','.join(list(map(str, absolute_location)))
		if len(markers) > 0:
			for marker in markers:
				r += '|' + str(marker.id)
				r += ',' + ','.join(list(map(str, list(marker.pos))))
				r += ',' + ','.join(list(map(str, list(marker.rot))))
		self.receiver.receive_result(r)


class ThreadManager(object):
	def __init__(self, width, height, cam_no):
		self.worker = DetectorThread('worker', width, height, cam_no, self)
		self.result_ready = False
		self.result = ""

	def start_thread(self):
		self.worker.start()

	def stop_thread(self):
		self.worker.stop()
		self.worker.join()

	def feed_image(self, feed):
		self.worker.feed_image(feed)

	def detect_markers(self):
		self.worker.detect_event.set()
		self.result_ready = False

	def receive_result(self, r):
		self.result = r
		self.result_ready = True

	def get_result(self):
		value = "NotReady"
		was_ready = self.result_ready
		self.result_ready = False

		if was_ready:
			value = self.result
		return value


