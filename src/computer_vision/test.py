import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = "../.."
IMAGES = f"{ROOT}/resources/images"


def main():
    imgbgr = np.array(cv2.imread(f"{IMAGES}/carousel.jpg"))

    img_rgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    plt.imshow(dilated)
    plt.show()


if __name__ == "__main__":
    main()