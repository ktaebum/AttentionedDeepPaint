import cv2
import numpy as np


def cv2_detect(filename):
    img = cv2.imread(filename, 0)

    edges = cv2.Canny(img, 100, 150, 3, L2gradient=True)

    return np.invert(edges)
