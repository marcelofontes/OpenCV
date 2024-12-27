import cv2
from ..config import config

class FaceTracker:
    """A tracker for facial features: face, eyes, nose, mouth."""
    
    def __init__(self, scaleFactor=1.2, minNeighbors =2, flags = cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        
        self._faces = []
        
        self._faceClassifier = cv2.CascadeClassifier(config.MODELS_PATH+'/Haar_cascade/Face_detection/haarcascade_frontalface_alt.xml')
        
        
    @property
    def faces(self):
        """The tracked facial features"""
        return self._faces