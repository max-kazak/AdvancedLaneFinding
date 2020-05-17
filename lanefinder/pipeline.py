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


class TopPerspectiveNode(PipeNode):

    def __init__(self, input, output, inv=False):
        self.input = input
        self.output = output
        self.inv = inv
        self.M, self.Minv = transform.calc_M(src=np.float32(transform.WARP_SRC_PTS),
                                             dst=np.float32(transform.WARP_DST_PTS))

    def _action(self, context):
        img = context.get(self.input)
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


class DisplayLaneFitNode(PipeNode):

    def __init__(self, lane_input, output):
        self.input = lane_input
        self.output = output

    def _action(self, context):
        lane = context.get(self.input)
        if lane is None:
            raise exceptions.PipeException("missing context parameter: {}".format(self.input))

        vis_img = lane.plot_fitted_lane()

        context[self.output] = vis_img

        return vis_img


def create_image_pipeline():
    return PipeLine('Image processing',
        [
            OverlayRoiNode(input='img',
                           output='img_roi'),
            CalibrationNode(input='img',
                            output='img'),
            ThresholdingNode(input='img',
                             output='binary'),
            CuttingNode(input='binary',
                        output='binary'),
            TopPerspectiveNode(input='binary',
                               output='binary_warped'),
            LaneDetectionNode(binary_input='binary_warped', prior_lane_input=None,
                              output='lane'),
            DisplayLaneFitNode(lane_input='lane',
                               output='fitted_lane_img')
        ])


def _main():
    pipeline = create_image_pipeline()
    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        context = {'img': img}
        final_res = pipeline.passthrough(context)
        if isinstance(final_res, np.ndarray):
            if len(final_res.shape) == 2:
                final_res = utils.mask_to_3ch(final_res)
            cv2.imshow("pipe result", final_res)
        else:
            print(final_res)

        cv2.imshow("input image", context['img_roi'])
        cv2.imshow("fitted lane", context['fitted_lane_img'])
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
