#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from os.path import join
from sys import exit

from os.path import join
from helpers import grayscale, gaussian_blur, canny, region_of_interest, \
    draw_lines, hough_lines, weighted_img, extrapolate, change_color
from utils import get_files

IN_DIR_IMAGES = '../test_images/'
IN_DIR_VIDEOS = '../'
OUT_DIR = './out/';
YELLOW_HSV_BOUNDARIES = ([20, 50, 50], [100, 255, 255])
WHITE = [255, 255, 255]


def pipeline(in_f, out_f = None):
    in_f = '../test_images/solidYellowCurve.jpg'

    img = mpimg.imread(in_f) if isinstance(in_f, str) else in_f
    color = change_color(img, YELLOW_HSV_BOUNDARIES, WHITE)
    gray = grayscale(color)
    blur_gray = gaussian_blur(gray, kernel_size=5)
    edges = canny(blur_gray, low_threshold=50, high_threshold=150)

    imshape = img.shape

    middle_x = math.ceil(imshape[1]/2)
    middle_y = math.ceil(imshape[0]/2)
    shift_x = math.ceil(0.05*imshape[1])
    shift_y = math.ceil(0.11*imshape[0])
    shift_y_visible = math.ceil(0.06*imshape[0])

    masked_edges = [
        # Left
        region_of_interest(
            edges,
            np.array([[
                (0, imshape[0]),
                (middle_x - shift_x, middle_y + shift_y),
                (middle_x, middle_y + shift_y),
                (middle_x, imshape[0]),
            ]], dtype=np.int32)
        ),
        # Right
        region_of_interest(
            edges,
            np.array([[
                (middle_x, imshape[0]),
                (middle_x, middle_y + shift_y),
                (middle_x + shift_x, middle_y + shift_y),
                (imshape[1], imshape[0])
            ]], dtype=np.int32)
        )
    ]
    pimage = img

    for index, masked_edge in enumerate(masked_edges):
        lines = hough_lines(masked_edge, rho=1, theta=np.pi/180, threshold=30, min_line_len=5, max_line_gap=10)

        #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #draw_lines(line_img, lines)

        lines = extrapolate(lines, y_from=middle_y + shift_y + shift_y_visible, y_to=imshape[0], left=(index==0))
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)

        pimage = weighted_img(line_img, pimage)

    if out_f:
        plt.imshow(pimage)
        plt.savefig(out_f)
    else:
        return pimage

def process_images(in_d, out_d):
    for f in get_files(in_d):
        in_f = join(in_d, f)
        out_f = join(out_d, f)
        pipeline(in_f, out_f)

def process_videos(in_d, out_d):
    for f in get_files(in_d, 'mp4'):
        in_f = join(in_d, f)
        out_f = join(out_d, f)
        clip1 = VideoFileClip(in_f)
        white_clip = clip1.fl_image(pipeline)
        white_clip.write_videofile(out_f, audio=False)


process_images(IN_DIR_IMAGES, OUT_DIR)
process_videos(IN_DIR_VIDEOS, OUT_DIR)
