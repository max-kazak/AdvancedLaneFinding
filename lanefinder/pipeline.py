"""
This module contains Pipeline class and Pipeline Node classes.
"""

import os
import logging

import numpy as np
import cv2

import paths
import calibration
import transform
import detect
import exceptions
import utils

log = logging.getLogger("lanefinder.pipeline")


class PipeLine:
    """
    Class that is comprised of PipeNodes and runs them sequentially.
    """

    def __init__(self, name, nodes):
        """
        :param name: pipeline name
        :param nodes: list of PipeNodes
        """
        self.name = name
        self.nodes = nodes

    def passthrough(self, context):
        """
        Pass context through pipeline nodes. Context contains both input parameters and intermediary results.

        :param context: dict of input arguments
        :return: last node results
        """
        log.info("Running {} Pipeline::".format(self.name))
        last_result = None
        for node in self.nodes:
            last_result = node.process(context)
        return last_result


class PipeNode:
    """
    Abstract class for all pipeline nodes.
    """

    def process(self, context):
        """
        Entrypoint to run the node.

        :param context: dict of input arguments
        :return: results of execution
        """
        log.info("Processing {}...".format(type(self).__name__))
        return self._action(context)

    def _action(self, context):
        """
        Override this method to implement custom node functionality.

        :param context: dict of input arguments
        :return:
        """
        raise RuntimeError('Called abstract method PipeNode._action()')


class ColorCvtNode(PipeNode):
    """
    Node that converts image from one color space to another.
    """

    RGB2BGR = cv2.COLOR_RGB2BGR
    BGR2RGB = cv2.COLOR_BGR2RGB

    def __init__(self, input, output, mode):
        """
        :param input: name of the input image in context
        :param output: name of the output image in context
        :param mode: cv2 compliant mode, see cv2.cvtColor
        """
        self.input = input
        self.output = output
        self.mode = mode

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        img_cvt = cv2.cvtColor(img, self.mode)
        context[self.output] = img_cvt
        return img_cvt


class CalibrationNode(PipeNode):
    """
    Undistort image using calibration matrices.
    """

    def __init__(self, input, output,
                 calibration_folder=paths.DIR_CAMERA_CAL, persist_to_file='calibration.p'):
        """
        :param input: name of the input image in context
        :param output: name of the output image in context
        :param calibration_folder: (optional) path to directory with calibration images
        :param persist_to_file: (optional) path to pickle file where calibration matrix is stored
        """
        self.input = input
        self.output = output
        self.mtx, self.dist = calibration.get_calibration(calib_img_path=calibration_folder,
                                                          pickle_path=persist_to_file)

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        img_cal = calibration.undistort(img, self.mtx, self.dist)
        context[self.output] = img_cal
        return img_cal


class ThresholdingNode(PipeNode):
    """
    Calculate binary mask.
    """
    
    def __init__(self, input, output):
        """
        :param input: name of the input image in context
        :param output: name of the output binary mask in context
        """
        self.input = input
        self.output = output

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        img_thresh = transform.combined_threshold(img)
        context[self.output] = img_thresh

        return img_thresh


class GrayThresholdingNode(PipeNode):
    """
    Apply thresholding to the grayscale image.
    """

    def __init__(self, gray_input, output, thresh):
        """
        :param gray_input: name of the input grayscale image in context
        :param output: name of the output binary mask in context
        :param thresh: threshold for pixel intensity
        """
        self.input = gray_input
        self.output = output
        self.thresh = thresh

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))

        img_thresh = np.zeros_like(img, dtype=np.uint8)
        img_thresh[img > self.thresh] = 1

        context[self.output] = img_thresh
        return img_thresh


class CuttingNode(PipeNode):
    """
    Removes pixels in the image outside of ROI
    """
    
    def __init__(self, input, output, roi_pts=transform.WARP_SRC_PTS):
        """
        :param input: name of the input image in context
        :param output: name of the output image in context
        :param roi_pts: (optinal) points of the ROI polygon
        """
        self.input = input
        self.output = output
        self.roi_pts = roi_pts

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        cut = transform.cut_roi(img, self.roi_pts)
        context[self.output] = cut

        return cut


class OverlayRoiNode(PipeNode):
    """
    Draws ROI on the image.
    """

    def __init__(self, input, output, roi_pts=transform.WARP_SRC_PTS):
        """
        :param input: name of the input image in context
        :param output: name of the output image in context
        :param roi_pts: (optional) points of the ROI polygon
        """
        self.input = input
        self.output = output
        self.roi_pts = roi_pts

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        img = img.copy()
        pts = np.array(self.roi_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        context[self.output] = img

        return img


class OverlayImagesNode(PipeNode):
    """
    Overlays one image on top the another
    """

    def __init__(self, bckgrnd_input, overlay_input, output, alpha=0.5):
        """
        :param bckgrnd_input: name of the bottom image in context
        :param overlay_input: name of the top image in context
        :param output: name of the output image in context
        :param alpha: (optional) transparency of the top image
        """
        self.input1 = bckgrnd_input
        self.input2 = overlay_input
        self.alpha = alpha
        self.output = output

    def _action(self, context):
        img1 = context.get(self.input1)
        if img1 is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input1))
        img2 = context.get(self.input2)
        if img2 is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input2))

        out_img = cv2.addWeighted(img1, 1, img2, self.alpha, 0)

        context[self.output] = out_img
        return out_img


class PerspectiveNode(PipeNode):
    """
    Applies perspective transformation to the image.
    """

    def __init__(self, input, output, inv=False, dtype=None):
        """
        :param input: name of the input image in context
        :param output: name of the output image in context
        :param inv: (default=False), inverse trasformation
        :param dtype: (optional) type of the output image
        """
        self.input = input
        self.output = output
        self.inv = inv
        self.M, self.Minv = transform.calc_M(src=np.float32(transform.WARP_SRC_PTS),
                                             dst=np.float32(transform.WARP_DST_PTS))
        self.dtype = dtype

    def _action(self, context):
        img = context.get(self.input)

        if self.dtype is not None:
            img = img.astype(self.dtype)

        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        if not self.inv:
            warped = transform.warpPerspective(img, self.M)
        else:
            warped = transform.warpPerspective(img, self.Minv)

        context[self.output] = warped

        return warped


class LaneDetectionNode(PipeNode):
    """
    Detects Lane on the binary mask image.
    """

    def __init__(self, binary_input, output, prior_lane_input=None):
        """
        :param binary_input: name of the binary input image in context
        :param prior_lane_input: (optional) name of the prior Lane object in context
        :param output: name of the output Lane object in context
        """
        self.input = binary_input
        self.prior_lane_input = prior_lane_input
        self.output = output

    def _action(self, context):
        binary_img = context.get(self.input)
        if binary_img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))

        prior_lane = context.get(self.prior_lane_input) if self.prior_lane_input is not None else None

        lane = detect.detect_llines(binary_img, prior_lane)

        context[self.output] = lane

        return lane


class DrawLaneNode(PipeNode):
    """
    Draw Lane representation.
    """

    def __init__(self, lane_input, output, mode='fit'):
        """
        :param lane_input: name of the input Lane object in context
        :param output: name of the output image in context
        :param mode: "fit" - draw fitted lines, "area" - draw lane area
        """
        self.input = lane_input
        self.output = output
        self.mode = mode

    def _action(self, context):
        lane = context.get(self.input)
        if lane is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))

        if self.mode == 'fit':
            vis_img = lane.plot_fitted_lane()
        elif self.mode == 'area':
            vis_img = lane.draw_lane()
        else:
            raise exceptions.PipeException("unsupported mode argument in DrawLaneNode: {}".format(self.mode))

        context[self.output] = vis_img

        return vis_img


class AddTextNode(PipeNode):
    """
    Add Lane information text to the image.
    """

    def __init__(self, lane_input, img_input, output):
        """
        :param lane_input: name of the input Lane object in context
        :param img_input: name of the input image in context
        :param output: name of the output image in context
        """
        self.output = output
        self.img_input = img_input
        self.lane_input = lane_input

    def _action(self, context):
        lane = context.get(self.lane_input)
        if lane is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.lane_input))
        img = context.get(self.img_input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.img_input))

        curv = lane.get_curv()

        offset = lane.get_offset()

        img_ar = img.copy()

        h = img_ar.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: {:4.0f}m'.format(np.absolute(curv))
        cv2.putText(img_ar, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

        offset_dir = 'right' if offset > 0 else 'left'
        text = '{:04.2f}m {} of center'.format(np.absolute(offset), offset_dir)
        cv2.putText(img_ar, text, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

        context[self.output] = img_ar
        return img_ar


def _create_lane_perception_pipeline():
    return PipeLine('Lane perception',
        [
            CalibrationNode(input='img',
                            output='img'),
            OverlayRoiNode(input='img',
                           output='img_roi'),
            # CuttingNode(input='img',
            #             output='img_cut'),
            # PerspectiveNode(input='img_cut',
            #                 output='calib_img'),
            ThresholdingNode(input='img',
                             output='binary'),
            CuttingNode(input='binary',
                        output='binary'),
            PerspectiveNode(input='binary', dtype=np.float32,
                            output='binary_warped'),
            GrayThresholdingNode(gray_input='binary_warped', thresh=0.8,
                                 output='binary_warped'),
            LaneDetectionNode(binary_input='binary_warped', prior_lane_input=None,
                              output='lane'),
            DrawLaneNode(lane_input='lane', mode='fit',
                         output='fitted_lane_img'),
            DrawLaneNode(lane_input='lane', mode='area',
                         output='lane_img'),
            PerspectiveNode(input='lane_img', inv=True,
                            output='lane_img'),
            OverlayImagesNode(bckgrnd_input='img', overlay_input='lane_img',
                              output='lane_ar'),
            AddTextNode(lane_input='lane', img_input='lane_ar',
                        output='lane_ar')
        ])


def _main():
    pipeline1 = _create_lane_perception_pipeline()
    img_dir = paths.DIR_TEST_IMG
    filenames = os.listdir(img_dir)
    for filename in filenames:
        img = cv2.imread(os.path.join(img_dir, filename))
        context = {'img': img}

        final_res = pipeline1.passthrough(context)

        cv2.imshow("input image", context['img_roi'])

        if isinstance(final_res, np.ndarray):
            if len(final_res.shape) == 2:
                final_res = utils.mask_to_3ch(final_res)
            cv2.imshow("pipe result", final_res)
        else:
            print(final_res)

        # cv2.imshow("fitted lane", context['fitted_lane_img'])
        cv2.waitKey()


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)20s | %(message)s')
    ch.setFormatter(formatter)

    root_logger = logging.getLogger('lanefinder')
    root_logger.addHandler(ch)
    root_logger.setLevel(logging.DEBUG)

    detect_logger = logging.getLogger('lanefinder.detect')
    detect_logger.setLevel(logging.INFO)

    _main()
