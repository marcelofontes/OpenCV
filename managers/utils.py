import cv2
import numpy
import scipy.interpolate


def isGray(image):
    """ Return True if the image has one channel per pixel"""
    return image.ndim < 3


def widthHeightDividedBy(image, divisor):
    """Return an image's dimensions, divided bu a value"""
    h, w = image.shape[:2]
    return (int(w/divisor),int(h/divisor))