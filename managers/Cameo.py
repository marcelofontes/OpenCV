import cv2
from managers.CaptureManager import CaptureManager
from managers.WindowManager import WindowManager
from .Trackers.trackers import HaarFaceTracker
from .Trackers.trackers import DlibHOGFaceTracker


cam1= cv2.VideoCapture(0) # camera embutida
cam2 =cv2.VideoCapture(2) # camera USb em /dev/video2

class Cameo(object):
    def __init__(self, faceTracker):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cam1, self._windowManager, True)
        self._faceTracker = faceTracker

        
    
    def run(self):
        """run the main loop"""
        self._windowManager.createWindow()
        
        while self._windowManager.isWindowCreated:
            
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            self._faceTracker.update(frame)
            faces =self._faceTracker.faces
            self._faceTracker.drawDebugRects(frame)
            # print(self._faceTracker.getFacesCoord())
            
            
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    
    def onKeypress(self, keycode):
        """handle a keypress
            space -> take a screeshot
            tab -> start/stop recording a screncast
            escape -> quit
        """
        
        if keycode == 32: #space
            self._captureManager.writeImage('screeshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape   
            self._windowManager.destroyWindow()


# if __name__ == "__main__":
#     Cameo().run()             
        
            
    