import cv2
from managers.CaptureManager import CaptureManager
from managers.WindowManager import WindowManager


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        
    
    def run(self):
        """run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
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
        
            
    