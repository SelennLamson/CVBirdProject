#!/usr/bin/env python3

import numpy as np
import cv2

def markerPosition(image, corners, quads, quad_height, mtx, dist):
    mtx = np.array([[6.5746697944293521e+002, 0, 500],
                    [0, 6.5746697944293521e+002, 375],
                    [0, 0, 1]])
    dist = np.array([[-4.1802327176423804e-001],
                     [5.0715244063187526e-001],
                     [0],
                     [0],
                     [-5.7843597214487474e-001]])

    def draw(img, corners, imgpts):
        '''
        X axis in blue color, Y axis in green color and Z axis in red color.
        '''
        meanx = 0
        meany = 0
        for i in range(4):
            meanx += corners[i][0]
            meany += corners[i][1]
        a = [round(meanx / 4).astype(int), round(meany / 4).astype(int)]
        quads = tuple(a)
        img = cv2.line(img, quads, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, quads, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, quads, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
        return img

    quad_corners = corners[quads.ravel(), :].reshape(quads.shape[0], 4, 2).astype(np.float32)

    # Position of each corner of a marker from its centre on x, y, z
    objp = np.array([[0, quad_height / 2, quad_height / 2], [0, -quad_height / 2, quad_height / 2],
                     [0, -quad_height / 2, -quad_height / 2], [0, quad_height / 2, -quad_height / 2]], dtype=np.float32)

    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3) * 30

    rotations = []
    translations = []
    for marker in quad_corners:
        ret, rvecs, tvecs = cv2.solvePnP(objp, marker, mtx, dist)
        imgpt, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #image1 = draw(img_markers, marker, imgpt)
        print(rvecs)
        print(tvecs)
        cv2_imshow(image1)
        rotations.append((rvecs, marker))
        translations.append((tvecs, marker))
    return rotations, translations