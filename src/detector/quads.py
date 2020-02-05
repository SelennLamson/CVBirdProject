import numpy as np

def detect_quads(bin_image, corners, samples=20, precision=0.95, min_dist=0.01, max_dist=1, orth_dst=0.05):
	"""
	From the binary image and the corners detected in it, finds convex quadrilaterals formed by them.
	:param bin_image: np(height, width), binary image (pure black and white pixels)
	:param corners: np(n_corners, 2), coordinates of the previously detected corners
	:param samples: number of samples to check per edge side
	:param precision: percentage of correct samples required to validate an edge
	:param min_dist: min edge size (percentage of image's biggest dimension)
	:param max_dist: max edge size (percentage of image's biggest dimension)
	:return: quads(n_quads, 4), numpy int arrays of corners' ids forming quads
	"""

	h = bin_image.shape[0]
	w = bin_image.shape[1]
	big_dim = max(w, h)
	min_dist_px = big_dim * min_dist
	max_dist_px = big_dim * max_dist
	nc = corners.shape[0]

	if nc < 4:
		return np.zeros((0, 4), dtype=int)

	# --- EDGES COMPUTATION ---
	# Start coordinate of each edge
	#   (nc, nc, 2)
	edges_start = corners[:, np.newaxis, :].repeat(nc, axis=1)

	# Vector of each edge from c1 to c2
	#   (nc, nc, 2)
	edges_vecs = edges_start.transpose(1, 0, 2) - edges_start

	# Distances values of each edge
	#   (nc, nc)
	edges_dists = np.sum(edges_vecs ** 2, axis=2) ** 0.5

	# Orthogonal direction vectors
	#   (nc, nc, 2)
	edges_ortho = np.dstack([-edges_vecs[:, :, 1], edges_vecs[:, :, 0]])

	# Sampling points in the picture, to check for edges, no lateral shift yet
	#   1D-array of how much we should move in terms of edge distance at each sample
	#   (samples)
	checks = (np.array(list(range(samples)), dtype=float) + 1) / (samples + 1)

	#   repeating along two dimensions for nc times
	#   (nc, nc, samples)
	checks = checks[np.newaxis, np.newaxis, :].repeat(nc, axis=0).repeat(nc, axis=1)

	#   multiplying by edge vectors, gaining one dimension at the end: (x, y)
	#   (nc, nc, samples, 2)
	checks = edges_vecs[:, :, np.newaxis, :] * checks[:, :, :, np.newaxis]

	#   adding the start vector of each edge
	#   (nc, nc, samples, 2)
	checks += edges_start[:, :, np.newaxis, :]

	# Shifting to the left and to the right of the edge
	#   (nc, nc, 2)
	lateral_shift = edges_ortho * orth_dst
	#   (nc, nc, samples, 2)
	checks_R = np.round(checks + lateral_shift[:, :, np.newaxis, :]).astype(int)
	checks_L = np.round(checks - lateral_shift[:, :, np.newaxis, :]).astype(int)

	# Clamping to image dimensions
	checks_R[:, :, :, 0] = np.clip(checks_R[:, :, :, 0], 0, w - 1)
	checks_R[:, :, :, 1] = np.clip(checks_R[:, :, :, 1], 0, h - 1)
	checks_L[:, :, :, 0] = np.clip(checks_L[:, :, :, 0], 0, w - 1)
	checks_L[:, :, :, 1] = np.clip(checks_L[:, :, :, 1], 0, h - 1)

	# Reading binary image at every sample point
	#   (nc, nc, samples)
	values_R = bin_image[checks_R[:, :, :, 1], checks_R[:, :, :, 0]].reshape(nc, nc, samples)
	values_L = bin_image[checks_L[:, :, :, 1], checks_L[:, :, :, 0]].reshape(nc, nc, samples)

	# Checking for each side of each edge if it passes the black/white threshold
	#   (nc, nc)
	values_R = (np.sum(values_R, axis=2) / samples)
	# whites_R = values_R > precision
	blacks_R = (1 - values_R) > precision

	values_L = (np.sum(values_L, axis=2) / samples)
	# whites_L = values_L > precision
	blacks_L = (1 - values_L) > precision

	# An edge is validated if both sides pass the threshold and they are not of the same color
	#   (nc, nc)
	# is_edge = np.bitwise_or(np.bitwise_and(whites_R, blacks_L), np.bitwise_and(blacks_R, whites_L))
	is_edge = np.bitwise_xor(blacks_L, blacks_R)

	# We filter edges that are too big or too small
	#   (nc, nc)
	is_edge &= np.bitwise_and(min_dist_px <= edges_dists, edges_dists <= max_dist_px)

	# We remove all edges above the diagonal of the matrix, to keep one-directional edges
	#   (nc, nc)
	is_edge = np.tril(is_edge, -1)

	# Edges are identified, we extract them.
	edges_c1, edges_c2 = np.where(is_edge)
	edges = np.array(list(zip(edges_c1, edges_c2)), dtype=int)

	if len(edges) > 0:
		edges_dists = edges_dists[edges[:, 0], edges[:, 1]]

	# --- QUADS COMPUTATION ---
	quads = []
	# prod_vecs = []

	# Every edge gets a chance to begin a quad as e1
	for i1 in range(len(edges)):
		e1 = edges[i1]
		e1c1 = e1[0]
		e1c2 = e1[1]

		# Checking for a second edge e2 that shares one corner with e1
		for i2 in range(i1 + 1, len(edges)):
			e2 = edges[i2]
			e2c1 = e2[0]
			e2c2 = e2[1]

			# If a corner is shared, we call:
			#   - ec1 : the shared corner
			#   - ec2 : the other corner from e2
			#   - ec4 : the other corner from e1 (because we will loop the quad here)
			if e1c1 == e2c1:
				ec1 = e1c1
				ec2 = e2c2
				ec4 = e1c2
			elif e1c1 == e2c2:
				ec1 = e1c1
				ec2 = e2c1
				ec4 = e1c2
			elif e1c2 == e2c1:
				ec1 = e1c2
				ec2 = e2c2
				ec4 = e1c1
			elif e1c2 == e2c2:
				ec1 = e1c2
				ec2 = e2c1
				ec4 = e1c1
			else:
				# No shared corner, we go to the next edge
				continue

			# We begin by looking for an edge that shares the other e1's corner (ec4)
			# We can start looking at i2+1 because every previous edge
			# doesn't share a corner with e1
			for i4 in range(i2 + 1, len(edges)):
				e4 = edges[i4]
				if ec4 not in e4:
					continue

				# We can deduce the third corner of the quad:
				#    - ec3 : the non-shared corner of e4
				e4c1 = e4[0]
				e4c2 = e4[1]
				ec3 = e4c1 if ec4 == e4c2 else e4c2

				# We finally look for an edge that links ec2 and ec3 (from e2 and e4)
				# We start looking at i1+1 because it could have been missed as it
				# doesn't have to share a corner with e1
				for i3 in range(i1 + 1, len(edges)):
					# Verify that it isn't e2 nor e4
					if i3 == i2 or i3 == i4:
						continue
					e3 = edges[i3]
					if ec2 not in e3 or ec3 not in e3:
						continue

					# Checking that smallest edge is not smaller than 20% of the biggest edge
					max_edge = max(edges_dists[i1], edges_dists[i2], edges_dists[i3], edges_dists[i4])
					min_edge = min(edges_dists[i1], edges_dists[i2], edges_dists[i3], edges_dists[i4])
					if min_edge < 0.3 * max_edge:
						continue

					v1 = edges_vecs[ec1, ec2]
					v2 = edges_vecs[ec2, ec3]
					v3 = edges_vecs[ec3, ec4]
					v4 = edges_vecs[ec4, ec1]
					v12 = np.sign(np.cross(v1, v2))
					v23 = np.sign(np.cross(v2, v3))
					v34 = np.sign(np.cross(v3, v4))
					v41 = np.sign(np.cross(v4, v1))
					if v12 == 0 or v23 == 0 or v34 == 0 or v41 == 0:
						continue
					if not v12 == v23 == v34 == v41:
						continue

					# The quad is validated, save its corners in the right order
					if v12 > 0:
						quads.append((ec1, ec2, ec3, ec4))
					else:
						quads.append((ec4, ec3, ec2, ec1))

	return np.array(quads, dtype=int)
