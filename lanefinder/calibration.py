import os
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import paths

log = logging.getLogger("lanefinder.calibration")


def _find_chessboard_corners(img, nx=9, ny=6, debug=False):
    """
    Finds chessboard corners on the img.

    :param img: BGR image of the chessboard
    :param nx: expected number of corners on the chessboard in x direction
    :param ny: expected number of corners on the chessboard in y direction
    :return: corner location if corners found, None otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners found
    if ret:
        # refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if log.getEffectiveLevel() == logging.DEBUG and debug:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('corners', img)
            cv2.waitKey()

        return corners_refined

    return None


def _calibrate(calib_img_path=paths.DIR_CAMERA_CAL, debug=False):
    """
    Obrains calibration matrices.

    :param calib_img_path: path to calibration images
    :return: mtx, dist if successful, None, None otherwise
    """
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    filenames_cal = os.listdir(paths.DIR_CAMERA_CAL)
    for filename_cal in filenames_cal:
        img = cv2.imread(os.path.join(paths.DIR_CAMERA_CAL, filename_cal))
        corners = _find_chessboard_corners(img)
        if corners is None:
            log.warning("error finding chessboard corners in {}".format(filename_cal))
        else:
            imgpoints.append(corners)
            objpoints.append(objp)

    test_img = cv2.imread(os.path.join(paths.DIR_CAMERA_CAL, filenames_cal[0]))
    img_size = (img.shape[1], img.shape[0])

    # Calculate calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if ret:
        if log.getEffectiveLevel() == logging.DEBUG and debug:
            test_img_undist = undistort(test_img, mtx, dist)
            cv2.imshow('distorted image', test_img)
            cv2.imshow('undistorted image', test_img_undist)
            cv2.waitKey()

        return mtx, dist

    return None, None


def get_calibration(calib_img_path=paths.DIR_CAMERA_CAL, pickle_path=None):
    """
    Obtains calibration matrices either from pickle file or by calculating them.

    :param calib_img_path: path to calibration images (default=paths.DIR_CAMERA_CAL)
    :param pickle_path: if passed and file exists then read calibration from the file,
                        if passed and file doesn't exist then calculate calibration and save to pickle file
    :return: mtx camera matrix, dist distortion matrix, (None,None) if calibration failed
    """
    if pickle_path is not None:
        # try to use pickle
        if os.path.exists(pickle_path) and os.path.isfile(pickle_path):
            # pickle exists
            log.info('reading calibration from {}'.format(pickle_path))
            dist_pickle = pickle.load(open(pickle_path, "rb"))
            mtx = dist_pickle["mtx"]
            dist = dist_pickle["dist"]
        else:
            # pickle doesn't exist
            mtx, dist = _calibrate(calib_img_path)
            if mtx is not None:
                log.info('saving calibration to {}'.format(pickle_path))
                dist_pickle = {"mtx": mtx, "dist": dist}
                pickle.dump(dist_pickle, open(pickle_path, "wb"))
    else:
        # don't use pickle
        mtx, dist = _calibrate(calib_img_path)

    return mtx, dist


def undistort(img, mtx, dist):
    """
    Applies undistortion to img.

    :param img: distorted image
    :param mtx: camera matrix
    :param dist: distortion matrix
    :return: undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def _test():
    _calibrate()
    #get_calibration(pickle_path='calibration.p')


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)

    _test()
