"""
Run this script to test different threshold levels.
"""

import os

import cv2
import numpy as np

import paths
import transform
import utils

def nothing(x):
    pass


filenames = os.listdir(paths.DIR_TEST_IMG)
images = []
for filename in filenames:
    images.append(cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename)))

# Create a black image, a window
cv2.namedWindow('controls')

# create trackbars for color change
cv2.createTrackbar('Hmin','controls',0,255,nothing)
cv2.createTrackbar('Hmax','controls',0,255,nothing)
cv2.createTrackbar('Smin','controls',0,255,nothing)
cv2.createTrackbar('Smax','controls',0,255,nothing)
cv2.createTrackbar('GMmin','controls',0,255,nothing)
cv2.createTrackbar('GMmax','controls',0,255,nothing)
cv2.createTrackbar('GDmin','controls',0,100,nothing)
cv2.createTrackbar('GDmax','controls',0,100,nothing)

# create switch for ON/OFF functionality
cv2.createTrackbar('orig/t_color/t_grad', 'controls',0,2,nothing)

while(1):
    # get current positions of four trackbars
    hmin = cv2.getTrackbarPos('Hmin','controls')
    hmax = cv2.getTrackbarPos('Hmax', 'controls')
    smin = cv2.getTrackbarPos('Smin', 'controls')
    smax = cv2.getTrackbarPos('Smax', 'controls')
    gmmax = cv2.getTrackbarPos('GMmax', 'controls')
    gmmin = cv2.getTrackbarPos('GMmin', 'controls')
    gdmax = cv2.getTrackbarPos('GDmax', 'controls')/100 * np.pi/2
    gdmin = cv2.getTrackbarPos('GDmin', 'controls')/100 * np.pi/2

    mode = cv2.getTrackbarPos('orig/t_color/t_grad', 'controls')

    # print(hmin, hmax, smin, smax)

    if mode == 0:
        for i, img in enumerate(images):
            cv2.imshow('image' + str(i), img)
    elif mode == 1:
        for i, img in enumerate(images):
            mask = transform.color_thresholding(img, thresh_h=(hmin, hmax), thresh_s=(smin, smax))
            cv2.imshow('image' + str(i), utils.mask_to_3ch(mask))
    elif mode == 2:
        for i, img in enumerate(images):
            mask = transform.grad_thresholding(img, thresh_mag=(gmmin, gmmax), thresh_dir=(gdmin, gdmax), kernel_size=5)
            cv2.imshow('image' + str(i), utils.mask_to_3ch(mask))

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
