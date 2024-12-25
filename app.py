import cv2
import argparse
import sys


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# inicia captura de video
source = cv2.VideoCapture(s)
# define janela de exibiçao
win_name='teste'
cv2.namedWindow=(win_name, cv2.WINDOW_NORMAL)

# loop de captura
while cv2.waitKey(1) != 27:
    # le a camera e captura os frames
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    # inverte imagem para ficar espelhada
    frame = cv2.flip(frame, 1)

    cv2.imshow(win_name,frame)

source.release()
cv2.destroyWindow(win_name)