#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import timeit
import math
import cv2

from scipy import interpolate

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def change_color(image, boundaries, color_to):
    b_from = np.array(boundaries[0], np.uint8)
    b_to = np.array(boundaries[1], np.uint8)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, b_from, b_to)
    try:
        result = image.copy()
        result[np.where(mask==0)] = 0
        result[np.where(mask!=0)] = color_to
    except:
        return image
    return weighted_img(image, result)

def filter_by_slope(lines, left, min_slope, max_slope):
    slopes = np.apply_along_axis(lambda r: (r[3] - r[1]) / (r[2] - r[0]), 2, lines)

    if left:
        slopes[slopes < -max_slope] = 0
        slopes[slopes > -min_slope] = 0
        lines = lines[np.where(slopes < 0)]
    else:
        slopes[slopes > max_slope] = 0
        slopes[slopes < min_slope] = 0
        lines = lines[np.where(slopes > 0)]

    return lines

def extrapolate(lines, y_from, y_to, left):
    if lines == None:
        return np.array([[]])

    lines_a = filter_by_slope(lines, left, .4, .8)
    x = np.reshape(lines_a[:, [0, 2]], (1, len(lines_a) * 2))[0]
    y = np.reshape(lines_a[:, [1, 3]], (1, len(lines_a) * 2))[0]

    try:
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
    except:
        return np.array([[]])

    def p(in_y):
        return (in_y - c)/m

    return np.array([[
        [
            math.ceil(p(y_from)),
            y_from,
            math.ceil(p(y_to)),
            y_to
        ]
    ]])

