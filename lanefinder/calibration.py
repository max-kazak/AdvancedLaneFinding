import os
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import paths

log = logging.getLogger("lanefinder.calibration")


def find_chessboard_corners(img, nx=9, ny=6):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners found
    if ret:
        if log.getEffectiveLevel() == logging.DEBUG:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('corners', img)
            cv2.waitKey()
        return corners
    else:
        return None


def main():
    for filename_cal in os.listdir(paths.DIR_CAMERA_CAL):
        img = cv2.imread(os.path.join(paths.DIR_CAMERA_CAL, filename_cal))
        corners = find_chessboard_corners(img)
        if corners is None:
            log.error("error finding chessboard corners in {}".format(filename_cal))


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)

    main()
