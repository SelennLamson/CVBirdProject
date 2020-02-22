# -*- coding: utf-8 -*-

class Marker():
    '''
    contains marker attributes
    '''
    
    def __init__(self, id_, corners=[], pos=None, rot=None):
        '''
        initialize a marker

        Parameters
        ----------
        id_ : int
            Aruco id of the marker
        corners : tuples of float, optional
            tuples (x, y) containing coordinate of corners (x4) in frame
        pos : float, optional
            position (x, y, z) from camera
        rot : float, optional
            rotation (xr, yr, zr) of marker

        Returns
        -------
        None.

        '''
        
        self.id = id_
        self.corners = corners.copy()
        self.pos = pos
        self.rot = rot
        
        
    def __str__(self):
        return f'id:{self.id}-pos:{self.pos}-rot:{self.rot}'