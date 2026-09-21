"""Traditional image-processing utilities for CS406."""
import cv2
import numpy as np

def _u8(image):
    a = np.asarray(image)
    return a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)

def gaussian(image):
    return cv2.GaussianBlur(_u8(image), (5, 5), 0)

def bilateral(image):
    return cv2.bilateralFilter(_u8(image), 9, 75, 75)

def sharpening(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(_u8(image), -1, kernel)

def sobel(image):
    gray = cv2.cvtColor(_u8(image), cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

def canny(image):
    gray = cv2.cvtColor(_u8(image), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

def hsv_histogram(image, bins=8):
    hsv = cv2.cvtColor(_u8(image), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [bins,bins,bins],
                        [0,180,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

TRAINABLE_PREPROCESS = {
    "gaussian": gaussian,
    "bilateral": bilateral,
    "sharpening": sharpening,
}
EDGE_ANALYSIS = {"sobel": sobel, "canny": canny}
