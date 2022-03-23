#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: GUI to create ROI's and mask the video.

:Author: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2021-11-21
"""
import cv2
import os
import json
from pathlib import Path
from tkinter import *

import numpy as np
from videoMask import roi_processing
from random import choice, seed, randint
from datetime import datetime
from tkinter.filedialog import askopenfilename
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def visualize_bbox(image: np.ndarray, tool, thickness: int = 2) -> np.ndarray:
    """
    Draws a bounding box on an image

    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
        style (str): False if rectangle without dashes have to be drawn
    Returns:
        image with a bounding box drawn on it.
    """
    start = (int(tool['bbox']["left"]), int(tool['bbox']["top"]))
    end = (int(tool['bbox']["left"] + tool['bbox']["width"]),
           int(tool['bbox']["top"] + tool['bbox']["height"]))
    h = tool['color'].lstrip('#')
    color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))  # BGR
    cv2.rectangle(image, start, end, color, thickness)
    cv2.putText(image, tool['title'][0] + '1', (start[0], start[1]-3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 1)

    return image


class App:
    def __init__(self, video, filepath1, filepath2=None, horizontal=True, w0=1880, h0=1021):
        # ------------------Drawing frames block------------------
        with open(filepath1, 'r') as f:
            self.file1 = json.load(f)

        self.horizontal = horizontal

        # ----------------OpenCV GUI block-----------------------
        self.w0, self.h0 = w0, h0
        self.vidpath = video
        self.vid = cv2.VideoCapture(video)
        _, self.frame = self.vid.read()
        self.windowName = 'image'
        self.trTitle = 'tracker'
        self.trackerPos = 0

        # ----------------Tkinter block---------------------------
        self.window = None  # child box
        self.wFrame = None  # frame in the child box
        self.rowFrames = []  # rows of wFrame with necessary intervals
        self.fName = None  # holds name of the Video to produce
        self.bg = None  # holds the chosen variant of coloring the background
        self.shape = None  # holds the chosen variant of ROI's shape
        self.custom = None  # if true then ROI should be changed on each frame (in progress)
        self.go2 = None  # holds the chosen parameter of the jump
        self.begin(w0, h0)

    def begin(self, w0, h0):
        """
        Starts
        + frame showing
        + creates a trackbar for navigation
        + set mouse callback to draw Roi
        + calls drawROI
        """
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        h, w = self.frame.shape[:2]

        k = 2 if self.horizontal else 1/2

        # dummy resizing in context of smart scaling. But is still dummy
        if h / h0 > k * w / w0:
            cv2.resizeWindow(self.windowName, int(k * w * h0 / h), h0)
        else:
            cv2.resizeWindow(self.windowName, self.w0, int(h * w0 / w / k))

        self.drawRoi(True)
        cv2.createTrackbar(self.trTitle, self.windowName, 0,
                           int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 2,
                           # -2 because we start from 0 and we show 2 frames at one time
                           self.trackbar)
        cv2.setMouseCallback(self.windowName, self.setRoi)
        cv2.waitKey(0)

    def setRoi(self, event, x, y, flags, params):
        """Mouse callback to set ROI

        event  - mouse event
        x: int  - x coordinate
        x: int  - y coordinate
        flags  - additional flags
        params - extra parameters
        """
        return None

    def drawRoi(self, flag=False):
        """
        Draws the set ROI and calls tkWidget function to change info in the child box
        flag (bool): helping tool to understand if it is necessary to add drawn ROIs to the child box
        """
        img1 = self.frame.copy()  # Copy image to not draw in over the original frame
        _, img2 = self.vid.read()
        cv2.putText(img1, str(self.trackerPos), (0, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
        cv2.putText(img2, str(self.trackerPos + 1), (0, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

        for obj in self.file1[self.trackerPos]['objects']:
            rt = max(1, int(img1.shape[1] / self.w0) + 1)
            img1 = visualize_bbox(img1, obj, rt)

        for obj in self.file1[self.trackerPos + 1]['objects']:
            rt = max(1, int(img2.shape[1] / self.w0) + 1)
            img2 = visualize_bbox(img2, obj, rt)

        if self.horizontal:
            img = np.hstack((img1, img2))
        else:
            img = np.vstack((img1, img2))

        cv2.imshow(self.windowName, img)

    def trackbar(self, val):
        """
        If trackbar changes position the new frame is shown on the screen
        val (str): new trackbar position
        """
        self.trackerPos = val
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(val))
        _, self.frame = self.vid.read()
        self.drawRoi()


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-v', '--video', type=str, help='path to video')
    parser.add_argument('-f1', '--filepath1', type=str, help='path to the MAL annotations')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-horizontal', type =bool, help="type of images' stack")
    group.add_argument('-vertical', type =bool, help="type of images' stack")

    parser.add_argument('-wsize', type=str, default="1600x1200", help='Your screen parameters WxH')
    opt = parser.parse_args()
    # "-v /home/valia/AntVideos/Cflo_troph_count_7-02_7-09.mp4 -f1 "
    # "/home/valia/AntVideos/Cflo_troph_count_masked_6-00_6-31_MAL.json "
    # "-vertical True".split())
    w, h = opt.wsize.split('x')
    flag = True if opt.horizontal else False
    App(opt.video, opt.filepath1, None, flag, int(w), int(h))
