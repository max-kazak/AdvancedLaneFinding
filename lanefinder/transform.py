import os
import logging

import numpy as np
import cv2

import paths
import utils

log = logging.getLogger("lanefinder.transform")

_src_top_mar = 550
_scr_bottom_mar = 170
_dst_top_mar = 400
_dst_bottom_mar = 500
WARP_SRC_PTS = [(_src_top_mar, 450),
                (1280 - _src_top_mar, 450),
                (1280 - _scr_bottom_mar, 690),
                (_scr_bottom_mar, 690)]
WARP_DST_PTS = [(_dst_top_mar, 0),
                (1280 - _dst_top_mar, 0),
                (1280 - _dst_bottom_mar, 720),
                (_dst_bottom_mar, 720)]
YM_PER_PX = 3 / 86
XM_PER_PX = 3.7 / 243


def grad_thresholding(img, thresh_abs_x=None, thresh_abs_y=None, thresh_mag=None, thresh_dir=None, kernel_size=3):
    """
    Create image mask using gradient thresholding.

    :param img: source bgr image
    :param thresh_abs_x: tuple(min,max), threshold using absolute x gradient value (0, 255)
    :param thresh_abs_y: tuple(min,max), threshold using absolute y gradient value (0, 255)
    :param thresh_mag: tuple(min,max), threshold using gradient magnitude value (0, 255)
    :param thresh_dir: tuple(min,max), threshold using gradient direction value (0, pi/2)
    :param kernel_size: sobel kernel size, odd number
    :return: binary mask of img size (1 channel)
    """
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
    """
    Crete image mask using color thresholding.

    :param img: source bgr image
    :param thresh_h: tuple(min,max), hue threshold in HSL color space
    :param thresh_s: tuple(min,max), saturation threshold in HSL color space
    :return: binary mask of img size (1 channel)
    """
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
    """
    Optimized combination of gradient and color masks to find lane lines on the road.

    :param img: source bgr image
    :param debug: show overlayed mask if log.getEffectiveLevel() == logging.DEBUG
    :return: combined binary mask of img size (1 channel)
    """
    mag = grad_thresholding(img, thresh_mag=(50, 200), kernel_size=5)
    dir = grad_thresholding(img, thresh_dir=(np.pi/9, np.pi/3), kernel_size=9)
    sat = color_thresholding(img, thresh_s=(80, 150))
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
    """
    Claculate perspective transform matrix.

    :param src: keypoints location on the source image.
    :param dst: keypooints location on the warped image
    :return: transform matrix M and inverse matrix Minv
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def cut_roi(img, poligon_pts):
    is_binary = False
    if len(img.shape) == 2:
        is_binary = True
        img = img.reshape((img.shape[0], img.shape[1], 1))
    mask = np.zeros_like(img, dtype=np.uint8)
    pts = np.array(poligon_pts, dtype=np.int32).reshape((-1, 1, 2))
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, [pts], ignore_mask_color)
    cut = cv2.bitwise_and(img, mask)
    if is_binary:
        cut = cut.reshape((img.shape[0], img.shape[1]))
    return cut


def warpPerspective(img, M):
    """
    Apply perspective transformation M to img.
    Note: replace M with Minv to unwarp image back.

    :param img: source image
    :param M: transformation matrix
    :return: warped image
    """
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def _test_warping(img, debug=True):
    # define source and destination points for transform
    src = WARP_SRC_PTS
    dst = WARP_DST_PTS

    M, Minv = calc_M(np.float32(src), np.float32(dst))

    img_warped = cut_roi(img, src)
    img_warped = warpPerspective(img_warped, M)

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        img_roi = img.copy()
        pts = np.array(src, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_roi, [pts], True, (0, 0, 255), 3)
        cv2.imshow('original_img', img_roi)
        cv2.imshow('warped', img_warped)
        cv2.waitKey()

    return img_warped


def _main():
    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        mask = combined_threshold(img)
        img_warped = _test_warping(utils.mask_to_3ch(mask))
        # img_warped = _test_warping(img)
        cv2.imwrite(os.path.join(paths.DIR_TEST_IMG_WARPED_LANES, filename), img_warped)


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('lanefinder').addHandler(ch)
    log.setLevel(logging.DEBUG)

    _main()
