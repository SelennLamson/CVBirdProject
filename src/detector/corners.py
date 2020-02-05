import cv2
import numpy as np

def draw_corners(image, corners):
	"""
	Draws dots on the image where corners where detected
	:param image: image to draw on
	:param corners: corners' coordinates
	:return: new image with corners drawn in blue (or white if grayscale)
	"""
	image_return = image.copy()

	if image.shape[2] == 3:
		color = (0, 0, 255)
	else:
		color = 255

	for i, corner in enumerate(corners):
		x,y = corner
		cv2.circle(image_return,(x,y),3,color,-1)

	return image_return


def identify_corners(image, binary, scale, mask=None):
	"""
	From a grayscale image, a binary image and an optional binary mask, identify corners through Harris corner detection.
	Corners are then filtered to select only those that are from a convex black shape.
	:param image: Source grayscale image
	:param binary: Source binary image
	:param scale: Scale of Harris detection
	:param mask: Optional binary mask
	:return: a numpy array of shape (n_corners, 2) containing the corners coordinates
	"""

	# Detecting corners on the image, allowing a lot of corners close to each-other but with high quality
	corners = cv2.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.01, minDistance=scale//2, mask=mask, useHarrisDetector=True)
	corners = np.int0(corners).reshape((corners.shape[0], 2))

	# We are looking for corners that are 1/4 black and 3/4 white, but perspective can deform them
	# Therefore, we select corners that are more than 55% white and less than 90% white at the same time
	true_corners = []
	for i, corner in enumerate(corners):
		x, y = corner
		if not(scale//2 < x < image.shape[1] - scale//2 and scale//2 < y < image.shape[0] - scale//2):
			continue
		if 0.9 > np.average(binary[y - scale:y + scale + 1, x - scale:x + scale + 1]) > 0.61:
			true_corners.append([x, y])

	return np.array(true_corners)



