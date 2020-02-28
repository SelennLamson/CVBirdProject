
import threading
import time
from typing import List

from src.model.Marker import Marker

class DetectorThread(threading.Thread):
	def __init__(self, name, w, h, receiver):
		super(DetectorThread, self).__init__(name=name)
		self.stop_event = threading.Event()

		self.w = w
		self.h = h
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
			time.sleep(0.01)

	def stop(self):
		self.stop_event.set()

	def feed_image(self, feed):
		flen = len(feed)
		if self.head + flen >= self.imlen:
			self.image[self.head:] = feed[:self.imlen - self.head]
		else:
			self.image[self.head : self.head + flen] = feed

		self.head += flen
		if self.head >= self.imlen:
			self.head = 0

	def detect_markers(self):
		# Preprocessing
		gray = self.image.copy().reshape(self.h, self.w)  # Conversion to gray scale

		markers, elapsed = detect_markers(gray, self.params)

		absolute_location = [0.0, 0.0, 0.0]

		# Building result string:
		# --> 'Results:X,Y,Z|101,x,y,z,rx,ry,rz|3,x,y,z,rx,ry,rz'
		r = 'Results:'
		r += ','.join(list(map(str, absolute_location)))
		for marker in markers:
			r += '|' + ','.join(list(map(str, [marker.id] + list(marker.pos) + list(marker.rot))))

		self.receiver.receive_result(r[:-1])


class ThreadManager(object):
	def __init__(self, width, height):
		self.worker = DetectorThread("worker", width, height, self)
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
		self.worker.detect_markers()

	def receive_result(self, r):
		self.result = r
		self.result_ready = True

	def get_result(self):
		was_ready = self.result_ready
		self.result_ready = False

		if was_ready:
			return self.result
		else:
			return "NotReady"


