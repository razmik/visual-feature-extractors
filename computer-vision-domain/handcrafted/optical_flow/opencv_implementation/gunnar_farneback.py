'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video


def draw_flow(img, flow, step=16):
    """
    Draw the optical flow on a mesh grid
    :param img: Video frame
    :param flow: optical flow: movement of each pixel
    :param step: number of distance between 2 points on the mesh grid (in pixel)
    :return: Frame with meshgrid and movement in lines.
    """
    color_of_dots = (0, 255, 0)  # Green
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)  # (480/16) * (640/16)
    fx, fy = flow[y, x].T  # Get the flow of each point defined as the mesh grid
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, color_of_dots)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, color_of_dots, -1)
    return vis


def draw_hsv(flow):
    """
    What is HSV?
    HSV is so named for three values—Hue, Saturation and Value.
    This color space describes colors (hue or tint) in terms of their
        shade (saturation or amount of gray) and
        their brightness value.
    Hue is expressed as a number from 0 to 360 degrees representing hues of
        red (which start at 0),
        yellow (starting at 60),
        green (starting at 120),
        cyan (starting at 180),
        blue (starting at 240) and
        magenta (starting at 300).
    Saturation is the amount of gray from zero percent to 100 percent in the color.
    Value (or brightness) works in conjunction with saturation and describes the brightness
    or intensity of the color from zero percent to 100 percent.

    Source: https://www.thoughtco.com/what-is-hsv-in-design-1078068

    H - represents the angle of movement, i.e. direction
    S - default value - 255
    V - intensity (brightness) of the colour corresponds to the magnitude of the movement.
    """

    # Manual method
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    mag = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(mag * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    import sys

    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    # Setting up screens
    # cv2.namedWindow("Optical Flow", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Optical Flow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_from_video = False
    if show_from_video:
        filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/video/Sequence_6/visor_1246523090489_new_6_camera1.avi".replace('\\', '/')
        filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/video\AVSS/AVSS_AB_Easy.avi".replace('\\', '/')
        fn = filepath

    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        """
        calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
        https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html
        
        What is this output?
        
        We get a 2-channel array with optical flow vectors, (u,v)
        So what you are actually getting is a matrix that has the same size as your input frame.
        Each element in that flow matrix is a point that represents the displacement of that pixel from the  prev frame. 
        Meaning that you get a point with x and y values (in pixel units) that gives you the delta x and delta y from 
        the last frame.
        """
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('Optical Flow', draw_flow(gray, flow, step=16))
        if show_hsv:
            cv2.imshow('Optical Flow in HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('Warping', cur_glitch)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
    cv2.destroyAllWindows()
