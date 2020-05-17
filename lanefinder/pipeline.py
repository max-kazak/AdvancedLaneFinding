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

    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    def passthrough(self, context):
        log.info("Running {} Pipeline::".format(self.name))
        last_result = None
        for node in self.nodes:
            last_result = node.process(context)
        return last_result


class PipeNode:

    def process(self, context):
        log.info("Processing {}...".format(type(self).__name__))
        return self._action(context)

    def _action(self, context):
        raise RuntimeError('Called abstract method PipeNode._action()')


class CalibrationNode(PipeNode):

    def __init__(self, input, output,
                 calibration_folder=paths.DIR_CAMERA_CAL, persist_to_file='calibration.p'):
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
    
    def __init__(self, input, output):
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

    def __init__(self, gray_input, output, thresh):
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
    
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def _action(self, context):
        img = context.get(self.input)
        if img is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))
        cut = transform.cut_roi(img, transform.WARP_SRC_PTS)
        context[self.output] = cut

        return cut


class OverlayRoiNode(PipeNode):

    def __init__(self, input, output, roi_pts=transform.WARP_SRC_PTS):
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

    def __init__(self, bckgrnd_input, overlay_input, output, alpha=0.5):
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

    def __init__(self, input, output, inv=False, dtype=None):
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

    def __init__(self, binary_input, prior_lane_input=None, output=None):
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

    def __init__(self, lane_input, output, mode='fit'):
        """

        :param lane_input:
        :param output:
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


def _create_lane_perception_pipeline():
    return PipeLine('Lane perception',
        [
            CalibrationNode(input='img',
                            output='img'),
            OverlayRoiNode(input='img',
                           output='img_roi'),
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
                              output='lane_ar')
        ])


def _main():
    pipeline1 = _create_lane_perception_pipeline()
    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        context = {'img': img}

        final_res = pipeline1.passthrough(context)

        if isinstance(final_res, np.ndarray):
            if len(final_res.shape) == 2:
                final_res = utils.mask_to_3ch(final_res)
            cv2.imshow("pipe result", final_res)
        else:
            print(final_res)

        cv2.imshow("input image", context['img_roi'])
        # cv2.imshow("fitted lane", context['fitted_lane_img'])
        cv2.waitKey()


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)20s | %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('lanefinder').addHandler(ch)
    log.setLevel(logging.DEBUG)

    detect_logger = logging.getLogger('lanefinder.detect')
    detect_logger.setLevel(logging.INFO)

    _main()
