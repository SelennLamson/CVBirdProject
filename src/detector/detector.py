import matplotlib.pyplot as plt
import time
import cv2

from src.detector.preprocessing import *
from src.detector.corners import *
from src.detector.quads import *
from src.detector.markers import *

from src.model.Marker import *
from scipy.spatial.transform import Rotation as R
#from src.model.Frame import Frame
import numpy as np


class DetectorParameters:
    def __init__(self):
        # Preprocessing & corners
        self.max_dim = None  # Max dimension, the other will be rescale
        self.apply_filter = None  # None, bilinear, gaussian
        self.binary_mode = binary_average  # Binarization function (see preprocessing.py)
        self.binary_mask = None  # Function to use to generate a mask for corner detection
        self.corners_scale = 10  # Window in which corners are searched (in pixels)
        self.max_corners = 500  # Maximum number of corner to detect
        self.corners_quality = 0.01  # Minimum treshold for a valid corner

        # Edges
        self.edge_samples = 20  # Number of samples taken along edges to validate them
        self.edge_precision = 0.95  # Percentage of samples that must agree to validate an edge
        self.edge_min_dist = 0.01  # Minimum length of an edge (in percentage of frame's biggest dimension)
        self.edge_max_dist = 0.8  # Maximum length of an edge (in percentage of frame's biggest dimension)
        self.orthogonal_dist = 0.05  # Distance that we walk away from the edge to take samples on both sides (in percentage of edge length)

        # Markers
        self.n_bits = 4  # Number of bits on the side of the markers' content
        self.border = 2  # Width of the markers' border, in bits
        self.aruco_dict = cv2.aruco.DICT_4X4_250  # Aruco dictionnary to use (should be consistent with n_bits)
        self.error_border = 0.1  # Percentage of error allowed on the border
        self.error_content = 0.05  # Percentage of error allowed on the content
        self.quad_height = 133.33 # height in real world of the markers

        # Debug
        self.draw_preprocessed = False  # Should the detector draw the preprocessed image as debug
        self.draw_binary = False  # Should the detector draw the binary image as debug
        self.draw_mask = False  # Should the detector draw the binary mask as debug
        self.draw_corners = False  # Should the detector overlay detected edges on image
        self.draw_quads = False  # Should the detector overlay detected quads on image
        self.return_preview = False  # Should the detector return the debug images, instead of plotting them directly


def detect_markers(src_img, params: DetectorParameters):
    """
	Locate markers on an RGB image and optionnaly displays an image with markers' perimeter highlighted and detected corners
	:param src_img:
	:param params:
	:return:
	"""

    start_time = time.time()

    resized = resize(src_img, params.max_dim)
    grayscale, binary, mask = preprocess(resized, apply_filter=params.apply_filter, binary_mode=params.binary_mode,
                                         binary_mask=params.binary_mask)
    corners = identify_corners(grayscale, binary, params.corners_scale, params.max_corners, params.corners_quality,
                               mask=mask)
    quads = detect_quads(binary, corners, samples=params.edge_samples, precision=params.edge_precision,
                         min_dist=params.edge_min_dist, max_dist=params.edge_max_dist, orth_dst=params.orthogonal_dist)
    bin_mats = extract_binary_matrices(binary, corners, quads, n_bits=params.n_bits + params.border * 2)
    indices, orientations = binary_check(bin_mats, cv2.aruco.Dictionary_get(params.aruco_dict), params.n_bits,
                                         params.border, params.error_border, params.error_content)
    markersList = compute_all_markers_position(corners, quads, indices, orientations, quad_height=params.quad_height)

    elapsed = time.time() - start_time

    if not (params.draw_preprocessed or
            params.draw_binary or
            params.draw_mask or
            params.draw_corners or
            params.draw_quads or
            params.return_preview):
        return markersList, elapsed

    if params.draw_mask and mask is not None:
        preview = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
    elif params.draw_binary:
        preview = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)
    elif params.draw_preprocessed:
        preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    else:
        preview = resized.copy()

    if params.draw_corners:
        preview = draw_corners(preview, corners)

    if params.draw_quads:
        for i, c in enumerate(quads):
            c1, c2, c3, c4 = c

            cv2.line(preview, tuple(corners[c1]), tuple(corners[c2]), (255, 0, 0), 2)
            cv2.line(preview, tuple(corners[c2]), tuple(corners[c3]), (255, 0, 0), 2)
            cv2.line(preview, tuple(corners[c3]), tuple(corners[c4]), (255, 0, 0), 2)
            cv2.line(preview, tuple(corners[c4]), tuple(corners[c1]), (255, 0, 0), 2)

            cv2.circle(preview, tuple(corners[c[orientations[i]]]), 2, (255, 255, 0), 3)

    if params.return_preview:
        return markersList, elapsed, preview

    plt.figure(figsize=(10, 7))
    plt.imshow(preview)
    plt.show()

    return markersList, elapsed


def compute_all_markers_position(corners, quads, indices, orientations, quad_height=77):
    """
    Compute the markers position in space : process only valid markers
    Parameters
    ----------
    corners : coordinates of the corners
    quads : identification of quads : 4 indexes tuples referring corner coordinates
    indices : indices of the marker (Aruco range). -1 if it's not a marker
    orientations : orientation of the marker. Encoding ??
    quad_height : the height of the quad in real world

    Returns
    -------
    List<Markers> the markers for the corresponding frame
    """

    # initiate the marker list
    markersList = []
    ids = []

    # iterate on quads
    for quad, marker_id, orientation in zip(quads, indices, orientations):
        # To avoid adding a marker with the same ID twice
        if marker_id in ids:
            continue
        else:
            ids.append(marker_id)

        # if the quad is a valid marker, eg, indice <> -1, identify it's location
        if marker_id > -1:
            # from corner index in quads, retrieve the coordinates
            quads_coordinates = np.array([corners[i] for i in quad], dtype=np.float32)
            # compute location and get a Marker object
            marker = compute_a_marker_position(marker_id, quads_coordinates, quad_height, orientation)
            markersList.append(marker)
    return markersList


def compute_a_marker_position(markerId, quad, quad_height, orientation):
    """
    for a given marker, compute it's location in space
    Parameters
    ----------
    markerId : the id of the marker
    quad : coordinate of the marker in image
    orientations : TBD
    quad_height : height of the marker in real world

    Returns
    -------

    """
    mtx = np.array([[209.69779199, 0., 299.50762478],
                    [0., 209.73435832, 298.8423724],
                    [0., 0.,1.]])
    dist = np.array([[1.30824056e-03, -2.92691888e-03, -2.45282270e-04, -9.55173831e-06, 1.38489429e-03]])

    mtx1 = np.array([[214.39165339, 0., 302.52855448],
                    [0., 214.75900295, 299.10275073],
                    [0., 0., 1.]])
    dist1 = np.array([[6.01735946e-04, -7.20592910e-04, -8.77851932e-05, 3.71261700e-04, -2.85035690e-04]])
    #objp = np.array([[0, -quad_height / 2, quad_height / 2], [0, quad_height / 2, quad_height / 2],
    #                 [0, quad_height / 2, -quad_height / 2], [0, -quad_height / 2, -quad_height / 2]],
    #                dtype=np.float32)


    # Marker's corners coordinates in its reference system
    objp = np.array([[-quad_height / 2, -quad_height / 2, 0], [quad_height / 2, -quad_height / 2, 0],
                     [quad_height / 2, quad_height / 2, 0], [-quad_height / 2, quad_height / 2, 0]],
                    dtype=np.float32)

    # Computing the rotation to apply to marker-space points, to reflect the found orientation
    total_rotation = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]) # This is identity matrix
    rotation_90z = np.array([[0, 1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]]) # This is 90 degrees rotation around z matrix
    for _ in range(orientation):
        total_rotation = rotation_90z @ total_rotation # We apply a 0/90/180/270 degrees rotation around z depending on marker orientation

    # Find transformation vectors between camera frame and marker frame
    ret, rvecs, tvecs = cv2.solvePnP((total_rotation @ objp.T).T, quad, mtx, dist)

    # Convert obtained rotation vectors to UE4 euler angles
    marker_rotation_matrix, jacobian = cv2.Rodrigues(rvecs)
    ue4_marker_rotation = R.from_matrix(marker_rotation_matrix)
    ue4_marker_eulers = ue4_marker_rotation.as_euler('zxy', degrees=True)

    # Build the translation vector transformed to UE4 coordinates
    transformation_matrix = np.array([[0, 0, 1],
                                      [1, 0, 0],
                                      [0, -1, 0]])
    ue4_location = np.dot(transformation_matrix, tvecs)

    return Marker(markerId, quad, ue4_location.reshape(-1), ue4_marker_eulers.reshape(-1))
