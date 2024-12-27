import cv2
from managers import utils
from managers import rects

from config import config


class Face(object):
    """ Data on facial feactures: face"""
    def __init__(self):
        self.faceRect = None
        

class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth."""
    
    def __init__(self, scaleFactor=1.2, minNeighbors =2, flags = cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        
        self._faces = []
        
        self._faceClassifier = cv2.CascadeClassifier(config.MODELS_PATH+'Haar_cascade/Face_detection/haarcascade_frontalface_default.xml')
        
        
    @property
    def faces(self):
        """The tracked facial features"""
        return self._faces
    
    def update(self, image):
        """upgrade the tracked face"""
        self._faces = []
        
        if utils.isGray(image):
            image=cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)
            
        minSize =utils.widthHeightDividedBy(image,8)
        faceRects = self._faceClassifier.detectMultiScale(image, self.scaleFactor, self.minNeighbors, self.flags,minSize=minSize)    
        
        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                face.faceRect = faceRect
                self._faces.append(face)
    
    
    def drawDebugRects(self,image):
        """Draw rectangles around the tracked face"""
        
        if utils.isGray(image):
            faceColor= 255
        else:
            faceColor=(0,255,255)
        
        for face in self.faces:
            rects.outlineRect(image, face.faceRect,faceColor)
    
    
    def getFacesCoord(self):
        
        if self.faces:
            
            x, y, w, h = self.faces[0].faceRect
            return (x, y, w, h)