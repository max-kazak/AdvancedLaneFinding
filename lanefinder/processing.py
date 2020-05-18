import os
import logging
import collections

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
    preproc_pipeline = PipeLine('Lane perception', [
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
        LaneDetectionNode(binary_input='binary_warped', prior_lane_input='prior_lane',
                          output='lane'),
    ])
    postproc_pipeline = PipeLine('Lane Augmented Reality', [
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

    lane_stack = collections.deque(maxlen=2)
    last_good_lane = None

    def process_frame(frame):
        # point to outside vars
        nonlocal lane_stack
        nonlocal last_good_lane

        if len(lane_stack) > 0:
            prior_lane = lane_stack[0]
        else:
            log.warning('no more prior lanes left, starting over...')
            prior_lane = None

        context = {'img': frame, 'prior_lane': prior_lane}
        preproc_pipeline.passthrough(context)

        lane = context['lane']
        display_lane = last_good_lane if last_good_lane is not None else lane
        if not lane.validate(prior_lane=prior_lane):
            log.warning('new lane is too bad, keep using last good lane')
            if len(lane_stack) > 0:
                lane_stack.pop()  # rm one lane from the stacks bottom
        else:
            # smooth lane
            lanefits = [lane.lane_fit for lane in lane_stack]
            lanefits.append(lane.lane_fit)
            display_lane = lane.copy()
            display_lane.lane_fit = _smooth_lane_fit(lanefits)
            last_good_lane = display_lane

            lane_stack.appendleft(lane)

        context['lane'] = display_lane  # lane to display
        postproc_pipeline.passthrough(context)

        return context['lane_ar']

    video_input = VideoFileClip(in_file)#.subclip(555/25, 560/25)
    video_processed = video_input.fl_image(process_frame)
    video_processed.write_videofile(out_file, audio=False)


def _smooth_lane_fit(lanefits):
    llinefits = np.array([lanefit[0] for lanefit in lanefits])
    rlinefits = np.array([lanefit[1] for lanefit in lanefits])
    llinefit_avg = np.mean(llinefits, axis=0)
    rlinefit_avg = np.mean(rlinefits, axis=0)
    return (llinefit_avg, rlinefit_avg)


def main():
    process_images(paths.DIR_TEST_IMG, paths.DIR_OUTPUT_IMG)
    process_video(os.path.join(paths.DIR_VIDEOS, paths.FILE_VIDEO_1),
                  os.path.join(paths.DIR_OUTPUT_VIDEOS, paths.FILE_VIDEO_1))
    # process_video(os.path.join(paths.DIR_VIDEOS, paths.FILE_VIDEO_2),
    #               os.path.join(paths.DIR_OUTPUT_VIDEOS, paths.FILE_VIDEO_2))
    # process_video(os.path.join(paths.DIR_VIDEOS, paths.FILE_VIDEO_3),
    #               os.path.join(paths.DIR_OUTPUT_VIDEOS, paths.FILE_VIDEO_3))


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)20s | %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('lanefinder').addHandler(ch)
    logging.getLogger('lanefinder').setLevel(logging.WARNING)

    main()
