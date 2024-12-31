import cv2
from managers import utils
from managers import rects
from config import config
import dlib
import abc
from ultralytics import YOLO



class Face(object):
    """ Data on facial feactures: face"""
    def __init__(self):
        self.faceRect = None
        
        
class FaceTracker(metaclass=abc.ABCMeta):
    """ Face tracker interface """
    
    @abc.abstractmethod
    def __init__(self):
        self._faces = []
    
    
    @property
    def faces(self):
        """The tracked facial features"""
        return self._faces


    @abc.abstractmethod
    def update(self, image):
        return


    @abc.abstractmethod
    def drawDebugRects(self,image):
        return


    def getFacesCoord(self):
        if self.faces:
            x, y, w, h = self.faces[0].faceRect
            return int(x), int(y), int(w), int(h)


class HaarFaceTracker(FaceTracker):
    """A Haar Cascade tracker for facial features: face."""
    
    def __init__(self, scaleFactor=1.2, minNeighbors =2, flags = cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._faceClassifier = cv2.CascadeClassifier(config.MODELS_PATH+'Haar_cascade/Face_detection/haarcascade_frontalface_default.xml')
        
    
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
    
    
class DlibHOGFaceTracker(FaceTracker):
    """A HOG  tracker for facial features: face."""

    def __init__(self):
        
        self._faceClassifier = dlib.get_frontal_face_detector()
        
 
    def update(self, image):
        """upgrade the tracked face"""
        
        self._faces = []
        faceRects = self._faceClassifier(image,1)
        
        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                x, y, w, h = faceRect.left(), faceRect.top(), faceRect.right()-faceRect.left(), faceRect.bottom()-faceRect.top() 
                face.faceRect = (x,y,w,h)
                self._faces.append(face)
    
    
    def drawDebugRects(self,image):
        """Draw rectangles around the tracked face"""
        
        faceColor=(0,255,255) # amarelo
        for face in self.faces:
            rects.outlineRect(image, face.faceRect,faceColor)
    

class CaffeFaceTracker(FaceTracker):
    """A Caffe Model based tracker for facial features: face."""
     
    def __init__(self, minConfidence=0.5):
        self._faceClassifier = cv2.dnn.readNetFromCaffe(prototxt=config.MODELS_PATH+"caffe/Face_detection/deploy .prototxt", 
                                                        caffeModel= config.MODELS_PATH+"caffe/Face_detection/res10_300x300_ssd_iter_140000.caffemodel")
        self._minConfidence = minConfidence
 
    def update(self, image):
        """upgrade the tracked face"""
        
        self._faces = []
        
        (height, width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)), scalefactor=1.0,
                                    size=(300, 300), mean=(104.0, 177.0, 123.0))
        
        self._faceClassifier.setInput(blob)
        faceRects = self._faceClassifier.forward()
        
        if faceRects is not None:
            for faceRect in faceRects[0][0]:
                confidence = faceRect[2]
                
                if confidence > self._minConfidence:
                    bbox = faceRect[3:]
                    face = Face()
                    x, y, w, h = int(bbox[0]*width), int(bbox[1]*height), int(abs(bbox[2]-bbox[0])*width), int(abs(bbox[3]-bbox[1])*height)
                    face.faceRect = (x,y,w,h)
                    self._faces.append(face)
        
    
    def drawDebugRects(self,image):
        """Draw rectangles around the tracked face"""
        
        faceColor=(0,255,255) # amarelo
        for face in self.faces:
            rects.outlineRect(image, face.faceRect,faceColor)


class YoloFaceTracker(FaceTracker):
    """A Yolo Model based tracker for facial features: face."""
     
    def __init__(self):
        self._faceClassifier = YOLO(config.MODELS_PATH+"Yolo/Face_detection/yolov8n-face.pt")
 
    def update(self, image):
        """upgrade the tracked face"""
        
        self._faces = []
        faceRects = self._faceClassifier.predict(source=image, show=False)
        
        if faceRects is not None:
            for faceRect in faceRects:
                box =faceRect.boxes.xyxy
                if len(box) >0:
                    face = Face()
                    x,y,w,h = int(box[0][0]), int(box[0][1]), int(box[0][2]) - int(box[0][0]), int(box[0][3])-int(box[0][1])
                    face.faceRect = (x,y,w,h)
                    self._faces.append(face)
        
    
    def drawDebugRects(self,image):
        """Draw rectangles around the tracked face"""
        
        faceColor=(0,255,255) # amarelo
        for face in self.faces:
            rects.outlineRect(image, face.faceRect,faceColor)


class YuNetFaceTracker(FaceTracker):
    """A YuNet Model based tracker for facial features: face."""
     
    def __init__(self):
        self._faceClassifier = cv2.FaceDetectorYN.create(config.MODELS_PATH+"YuNet/Face_detection/face_detection_yunet_2023mar.onnx",
                                                         "", (0, 0))
     
        
    def update(self, image):
        """upgrade the tracked face"""
        
        self._faces = []
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        height, width, _ = image.shape
        self._faceClassifier.setInputSize((width, height))
        _,faceRects =  self._faceClassifier.detect(image)
        faceRects = faceRects if faceRects is not None else []
        
        if faceRects is not None:
            for faceRect in faceRects:
                box = list(map(int, faceRect[:4]))
                
                if len(box) >0:
                    face = Face()
                    x,y,w,h = int(box[0]), int(box[1]), int(box[2]) , int(box[3])
                    face.faceRect = (x,y,w,h)
                    self._faces.append(face)
        
    
    def drawDebugRects(self,image):
        """Draw rectangles around the tracked face"""
        
        faceColor=(0,255,255) # amarelo
        for face in self.faces:
            rects.outlineRect(image, face.faceRect,faceColor)




 
