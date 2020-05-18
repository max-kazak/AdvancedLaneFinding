import os

import numpy as np
import cv2

import paths
from pipeline import *

log = logging.getLogger("lanefinder.processing")


def process_images():
    pipeline = PipeLine('Lane perception', [
            CalibrationNode(input='img',
                            output='img'),
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
            DrawLaneNode(lane_input='lane', mode='area',
                         output='lane_img'),
            PerspectiveNode(input='lane_img', inv=True,
                            output='lane_img'),
            OverlayImagesNode(bckgrnd_input='img', overlay_input='lane_img',
                              output='lane_ar'),
            AddTextNode(lane_input='lane', img_input='lane_ar',
                        output='lane_ar')
            ])

    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        log.info('Processing image {}'.format(filename))
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        context = {'img': img}
        pipeline.passthrough(context)
        cv2.imwrite(os.path.join(paths.DIR_OUTPUT_IMG, filename), context['lane_ar'])


def main():
    process_images()


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)20s | %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('lanefinder').addHandler(ch)
    log.setLevel(logging.DEBUG)

    detect_logger = logging.getLogger('lanefinder.detect')
    detect_logger.setLevel(logging.INFO)

    main()
