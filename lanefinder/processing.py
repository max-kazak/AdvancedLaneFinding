import os
import logging

import numpy as np
import cv2
from moviepy.editor import VideoFileClip

import paths
from pipeline import PipeLine, PerspectiveNode, CalibrationNode, ThresholdingNode, CuttingNode, GrayThresholdingNode, \
    LaneDetectionNode, DrawLaneNode, OverlayImagesNode, AddTextNode, ColorCvtNode

log = logging.getLogger("lanefinder.processing")


def process_images(in_dir, out_dir):
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

    filenames = os.listdir(in_dir)
    for filename in filenames:
        log.info('Processing image {}'.format(filename))
        img = cv2.imread(os.path.join(in_dir, filename))
        context = {'img': img}
        pipeline.passthrough(context)
        cv2.imwrite(os.path.join(out_dir, filename), context['lane_ar'])


def process_video(in_file, out_file):
    pipeline = PipeLine('Lane perception', [
        ColorCvtNode(input='img', mode=ColorCvtNode.RGB2BGR,
                     output='img'),
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
                    output='lane_ar'),
        ColorCvtNode(input='lane_ar', mode=ColorCvtNode.BGR2RGB,
                     output='lane_ar'),
    ])

    def process_frame(frame):
        # cv2.imwrite('debug/failed.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        context = {'img': frame}
        pipeline.passthrough(context)
        return context['lane_ar']

    video_input = VideoFileClip(in_file)#.subclip(555/25, 560/25)
    video_processed = video_input.fl_image(process_frame)
    video_processed.write_videofile(out_file, audio=False)


def main():
    process_images(paths.DIR_TEST_IMG, paths.DIR_OUTPUT_IMG)
    # process_video(os.path.join(paths.DIR_VIDEOS, paths.FILE_VIDEO_1),
    # os.path.join(paths.DIR_OUTPUT_VIDEOS, paths.FILE_VIDEO_1))



if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)20s | %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('lanefinder').addHandler(ch)
    logging.getLogger('lanefinder').setLevel(logging.WARNING)

    main()
