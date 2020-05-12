import os
import logging

import numpy as np
import cv2

import paths

log = logging.getLogger("lanefinder.transform")


def grad_thresholding(img, thresh_abs_x=None, thresh_abs_y=None, thresh_mag=None, thresh_dir=None, kernel_size=3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply x and y gradient with the OpenCV Sobel() function
    # and take the absolute value
    sobelX_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sobelY_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size))

    binary_combined = np.ones_like(gray)

    if thresh_abs_x is not None:
        # Apply abs x thresholding
        sobelX_scaled = np.uint8(255 * sobelX_abs / np.max(sobelX_abs))
        binary_abs_x = np.zeros_like(sobelX_scaled)
        binary_abs_x[(sobelX_scaled >= thresh_abs_x[0]) & (sobelX_scaled <= thresh_abs_x[1])] = 1
        binary_combined = binary_combined * binary_abs_x

    if thresh_abs_y is not None:
        # Apply abs y thresholding
        sobelY_scaled = np.uint8(255 * sobelY_abs / np.max(sobelY_abs))
        binary_abs_y = np.zeros_like(sobelY_scaled)
        binary_abs_y[(sobelY_scaled >= thresh_abs_y[0]) & (sobelY_scaled <= thresh_abs_y[1])] = 1
        binary_combined = binary_combined * binary_abs_y

    if thresh_mag is not None:
        # Apply magnitude thresholding
        # Calculate the gradient magnitude
        mag = np.sqrt(sobelX_abs ** 2 + sobelY_abs ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(mag) / 255
        mag_scaled = (mag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_mag = np.zeros_like(mag_scaled)
        binary_mag[(mag_scaled >= thresh_mag[0]) & (mag_scaled <= thresh_mag[1])] = 1
        binary_combined = binary_combined * binary_mag

    if thresh_dir is not None:
        # Apply gradient direction thresholding
        dir = np.arctan2(sobelY_abs, sobelX_abs)
        binary_dir = np.zeros_like(dir)
        binary_dir[(dir >= thresh_dir[0]) & (dir <= thresh_dir[1])] = 1
        binary_combined = binary_combined * binary_dir

    return binary_combined.astype(np.uint8)


def color_thresholding(img, thresh_h=None, thresh_s=None):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    binary_combined = np.ones_like(S)

    if thresh_s is not None:
        binary = np.zeros_like(S)
        binary[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
        binary_combined = binary_combined * binary

    if thresh_h is not None:
        binary = np.zeros_like(H)
        binary[(H > thresh_h[0]) & (H <= thresh_h[1])] = 1
        binary_combined = binary_combined * binary

    return binary_combined


def combined_threshold(img, debug=False):
    mag = grad_thresholding(img, thresh_mag=(50, 200), kernel_size=5)
    dir = grad_thresholding(img, thresh_dir=(np.pi/9, np.pi/3), kernel_size=9)
    sat = color_thresholding(img, thresh_s=(80, 255))
    hue = color_thresholding(img, thresh_h=(15, 100))

    mask = hue * sat + mag * dir

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        mask_3ch = np.zeros_like(img)
        mask_3ch[:, :, 2] = mask * 255
        overlay = cv2.addWeighted(img, 0.7, mask_3ch, 1., 0.)
        cv2.imshow('overlay', overlay)
        cv2.waitKey()

    return mask


def calc_M(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warpPerspective(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def mask_to_3ch(mask, r=True, g=True, b=True):
    mask_3ch = np.zeros((mask.shape[0], mask.shape[1], 3))
    if b:
        mask_3ch[:, :, 0] = mask * 255
    if g:
        mask_3ch[:, :, 1] = mask * 255
    if r:
        mask_3ch[:, :, 2] = mask * 255
    return mask_3ch


def _test_warping(img):
    # define source and destination points for transform
    src = np.float32([(570, 450),
                      (740, 450),
                      (1110, 670),
                      (220, 670)])
    dst = np.float32([(250, 50),
                      (1280 - 250, 50),
                      (1280 - 250, 720-10),
                      (250, 720-10)])

    M, Minv = calc_M(src, dst)

    img_warped = warpPerspective(img, M)

    if log.getEffectiveLevel() == logging.DEBUG:
        pts = src.reshape((-1, 1, 2)).astype(np.int32)
        img_roi = cv2.polylines(img, [pts], True, (0, 0, 255), 3)
        cv2.imshow('original_img', img_roi)
        cv2.imshow('warped', img_warped)
        cv2.waitKey()


def _main():
    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        mask = combined_threshold(img)
        _test_warping(img)


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)

    _main()
