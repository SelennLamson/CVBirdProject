#!/usr/bin/env python3

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.linalg import norm
import numpy as np
import json

def least_square_camera_problem(X, P1, P2):
    """
    Formulate the problem we want to minimize :
        SUM_OVER_N(NORM(X + P1 + P1)^2)

    Parameters
    ----------
    X : array of float (size 3)
        coordinate of the camera
    P1 : matrix of size N x 3
        Points location in frame 1
    P2 : matrix of size N x 3
        Points location in frame 2

    Returns
    -------
    SUM_OVER_N(NORM(X + Quads_frame_1 + Quads_frame_2)^2)

    """
    #print(X + P2 - P1)
    normed = norm(X + P2 - P1, axis = 1)
    #print(normed)
    
    return np.sum(normed)

#%%

def test_least_square_camera_problem():
    X = np.array([0, 0, 0])
    P1 = np.array([[1, 2, 0], [7, 8, 0], [7, 8, 0],[13, 8, 0]])
    P2 = np.array([[5, 6, 0], [10, 11, 0], [7, 8, 0], [6, 8, 0]])
    
    r = solve_LS_camera_problem(P1, P2, X)
    print(r)
    
#%%
    
def solve_LS_camera_problem(P1, P2, X0=[0, 0, 0]):
    """
    Resolve the minimisation problem:
        SUM_OVER_N(NORM(X + P1 + P1)^2)

    Parameters
    ----------
    P1 : matrix of size N x 3
        Points location in frame 1
    P2 : matrix of size N x 3
        Points location in frame 2
    X0 : initial guess for X, optimally, previous positions

    Returns
    -------
    X solution, None if no solution

    """
    res_lsq = least_squares(least_square_camera_problem, X0, args=(P1, P2))
    
    return res_lsq['x']
    
def test_trans_rot():
    P_front_m= [ -135.0302734375, -3.1635982990264893, -38.12091064453125 ]
    P_bottom_m = [ 15.010517120361328, -3.1635730266571045, -31.127927780151367 ]
    rotation_front_in_bottom = [ -47.423492431640625, 90, -47.423492431640625 ]
    P_front_bottom = [ -23.110374450683594, -2.8448926059354562e-06, 103.90232849121094 ]
    
    rot_mat = Rotation.from_rotvec(np.deg2rad(rotation_front_in_bottom)).inv()
    
    P_bottom_m_hat = rot_mat.as_matrix() @ P_front_m + P_front_bottom
    
    print(P_bottom_m_hat)
    

