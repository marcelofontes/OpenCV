from managers.Cameo import Cameo
from managers.Trackers.trackers import (DlibHOGFaceTracker, HaarFaceTracker, 
                                        CaffeFaceTracker, YoloFaceTracker,
                                        YuNetFaceTracker)

# faceTracker = DlibHOGFaceTracker() # Deteçao boa, mas performance ruim
# faceTracker = HaarFaceTracker()  #detecçao ruim, porém leve
# faceTracker =CaffeFaceTracker(minConfidence=0.5)  # detecçao boa e performance ótima
# faceTracker = YoloFaceTracker()  # detecçao boa e performance ótima, porém precis a Ultralytics que é grande para o raspberry
faceTracker = YuNetFaceTracker()  # detecçao boa e performance boa

camera = Cameo(faceTracker=faceTracker)
camera.run()
