import os
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt

import paths

log = logging.getLogger("lanefinder.detect")


def img_pipeline(img):
    pass


def _main():
    filenames = os.listdir(paths.DIR_TEST_IMG)
    for filename in filenames:
        img = cv2.imread(os.path.join(paths.DIR_TEST_IMG, filename))
        img_pipeline(img)


if __name__ == '__main__':
    # logging setup
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)

    _main()
