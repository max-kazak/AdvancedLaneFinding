import os
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt

import paths
import utils

log = logging.getLogger("lanefinder.detect")


def _find_lane_loc_hist(binary_warped, window_h=None, debug=True):
    """
    Find initial location of the lane lines on the bottom part of the image.

    :param binary_warped: binary or 0-1 mask of the lane lines
    :param window_h: height of the processing window
    :return: (leftx_base, rightx_base) - locations of the left and right lane lines
    """
    if window_h is None:
        window_h = binary_warped.shape[0] // 2

    histogram = np.sum(binary_warped[-window_h:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        fig, plot = utils.plot_for_img(binary_warped)
        plot.imshow(np.flip(utils.mask_to_3ch(binary_warped), axis=0), origin='lower')
        plot.plot(histogram)
        cv2.imshow('histogram', utils.fig2data(fig))
        cv2.waitKey()

    return leftx_base, rightx_base


def _segment_llines_noprior(binary_warped, nwindows=9, margin=100, minpix=100, debug=True):
    """
    Find pixels that belong to lane lines.

    :param binary_warped: input binary mask warped to top perspective
    :param nwindows: number of iterating windows
    :param margin: window width
    :param minpix: min number of pixels found to recenter window
    :return:
    """
    # find initial rough positions of the bottom lines using histogram
    leftx_base, rightx_base = _find_lane_loc_hist(binary_warped)

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        out_img = utils.mask_to_3ch(binary_warped)

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if log.getEffectiveLevel() == logging.DEBUG and debug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # except ValueError:
    #     # Avoids an error if the above is not implemented fully
    #     pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        cv2.imshow('window lane search', out_img)
        cv2.waitKey()

    return leftx, lefty, rightx, righty


def _segment_llines_prior(binary_warped, left_fit, right_fit, margin=100, debug=True):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        # Create an image to draw on and an image to show the selection window
        out_img = utils.mask_to_3ch(binary_warped)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        overlay = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        cv2.imshow('fit over prior', overlay)
        cv2.waitKey()

    return leftx, lefty, rightx, righty


def _fit_polynomial(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def detect_llines(binary_warped, prior=None, debug=True):
    if prior is None:
        leftx, lefty, rightx, righty = _segment_llines_noprior(binary_warped)
    else:
        leftx, lefty, rightx, righty = _segment_llines_prior(binary_warped, left_fit=prior[0], right_fit=prior[1])

    left_fit, right_fit = _fit_polynomial(leftx, lefty, rightx, righty)

    if log.getEffectiveLevel() == logging.DEBUG and debug:
        out_img = utils.mask_to_3ch(binary_warped)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        fig, plot = utils.plot_for_img(out_img)
        plot.plot(left_fitx, ploty, color='yellow')
        plot.plot(right_fitx, ploty, color='yellow')
        plot.imshow(out_img)
        plt_img = utils.fig2data(fig)
        cv2.imshow('fitted lines', plt_img)
        cv2.waitKey()

    return left_fit, right_fit


def _main():
    filenames = os.listdir(paths.DIR_TEST_IMG_WARPED_LANES)
    lane_fit = None
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG_WARPED_LANES, filename))
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask[img[:, :, 0] > 0] = 1

        detect_llines(mask, lane_fit)


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)

    _main()
