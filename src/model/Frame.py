# -*- coding: utf-8 -*-
import json
from  model.Marker import Marker
import numpy as np

class Frame():
    """
        This class contain frame information, expecially the markers identified on frame
    """
    
    def __init__(self, id_, markers = [], cam_rot=(0,0,0)):
        """
        Initialise the frame

        Parameters
        ----------
        id_ : int
            frame id (in the video).
        markers : [p1, p2, p3, p4], float , optional
            List of markers identified on the frame
        cam_rot: tuple of float
            Rotation of camera in degrees

        Returns
        -------
        None.

        """
        self.id = id_
        self.markers = markers.copy()
        self.camRot = np.array(cam_rot)
        
        
    def addMarkersAndRotFromJson(self, folder = '../data/simulation/labels_600x600/', view = 'front',
                                 addMarkers = False, filterVisible = True):
        """
        Add markers provided by a JSON file (simulation)

        Parameters
        ----------
        addMarkers : do we add the markers from the json
        folder : string
            folder where are store the JSON files.
        view : string
            'front' for front camera, 'rear' for rear one
        filterVisible : boolean
            True if we want to proceed only visible markers

        Returns
        -------
        None.

        """
        
        #open the JSON file
        with open(f'{folder}frame_{self.id}.json') as json_file:
            data = json.load(json_file)

        #access directly to cameras info
        cameras_data = list(data.values())[0]
        
        #select front or rear information
        cameraOfInterest = cameras_data[0] if view == 'front' else cameras_data[1]
        
        #get the list of visible markers are only them will be stored
        visibleMarkers = cameraOfInterest['visible']
        
        #iterate on markers, if visible add them to frame
        markersList = cameraOfInterest['markers']
        
        if addMarkers:
            for marker in markersList:
                if marker['id'] in visibleMarkers or (not filterVisible):
                    currMarker = Marker(marker['id'], pos = marker['location'],
                                        rot = marker['rotation'])
                    self.markers.append(currMarker)

        #processing the rotation
        self.camRot = np.array(cameraOfInterest['rotation'])
        
        #processing the location of the camera
        #only for validation purpose
        self.real_position_HIDDEN = np.array(cameraOfInterest['location'])

        
    def getMarkersId(self):
        '''
        return the markerIds contained in the frame as a Set
        '''
        ids = []
        for marker in self.markers:
            ids.append(marker.id)
            
        return set(ids)
    
    def getMatrixPos(self, markers_Id):
        '''
        for all markers in markers_Id return marker pos (marker.pos)
        as numpy matric

        Parameters
        ----------
        markers_Id : collection
            list of markers.

        Returns
        -------
        numpy array [Nx3] of positions

        '''
        result = []
        
        for marker in self.markers:
            if marker.id in markers_Id:
                result.append(marker.pos)
                
        return np.array(result)
        

def test_Frame():
    frame = Frame(36)
    frame.addMarkersAndRotFromJson(filterVisible=False, addMarkers=True)
    for mark in frame.markers:
        print(mark)
    print(frame.getMarkersId())
        