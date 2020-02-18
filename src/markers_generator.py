#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cv2 import aruco as aruco
import cv2

DICT_44_250 = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

#target markers will be from 0 to 9
MARKER_TARGET_RANGE = range(10)
#obstacle markers will be from 10 to 149
MARKER_OBSTACLE_RANGE = range(10, 150)
#other markers will be from 150 to 250
MARKER_OTHER =  range(150, 250)


def generate_markers(kind = 'target'):
    '''
    generate all markers for a given kind

    Parameters
    ----------
    kind : string, mandatory
        either 'target, 'obstacle', 'other'. The default is 'target'.

    Returns
    -------
    None.

    '''
    
    range_ = MARKER_TARGET_RANGE if kind == 'target' \
                else MARKER_OBSTACLE_RANGE if kind == 'obstacle' \
                else MARKER_OTHER
                
    
    #build the markers and save them to disk
    for idx in range_:
        curr_marker = aruco.drawMarker(dictionary=DICT_44_250, id=idx, sidePixels=200, borderBits=2)
        cv2.imwrite(f'../data/markers/marker_{kind}_{idx}.png', curr_marker)
        
def generate_all_markers():
    generate_markers('target')
    generate_markers('obstacle')
    generate_markers('other')