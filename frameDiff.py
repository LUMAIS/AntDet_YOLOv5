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


def visualize_bbox(image: np.ndarray, tool, style=False, thickness: int = 2) -> np.ndarray:
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
    if not style:
        cv2.rectangle(image, start, end, color, thickness)
        cv2.putText(image, tool['id'], (start[0], start[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    else:
        cv2.rectangle(image, start, end, color, round(thickness*1.5))
        cv2.putText(image, tool['id'], (start[0], start[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image


class App:
    def __init__(self, video, filepath1, filepath2=None, horizontal=True, w0=1880, h0=1021):
        # ------------------Drawing frames block------------------
        with open(filepath1, 'r') as f:
            self.file1 = json.load(f)

        # with open(filepath1.rstrip('json') + 'improved.json', 'w') as f:
        #     self.file2 = json.load(f)

        self.horizontal = horizontal

        # ----------------OpenCV GUI block-----------------------
        self.w0, self.h0 = w0, h0
        self.vidpath = video
        self.vid = cv2.VideoCapture(video)
        _, self.fframe = self.vid.read()
        _, self.nframe = self.vid.read()
        self.h, self.w = self.fframe.shape[:2]
        self.windowName = 'image'
        self.trTitle = 'tracker'
        self.trackerPos = 0

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
        h, w = self.fframe.shape[:2]

        k = 2 if self.horizontal else 1/2

        # dummy resizing in context of smart scaling. But is still dummy
        if h / h0 > k * w / w0:
            cv2.resizeWindow(self.windowName, int(k * w * h0 / h), h0)
        else:
            cv2.resizeWindow(self.windowName, self.w0, int(h * w0 / w / k))

        self.drawRoi()
        cv2.createTrackbar(self.trTitle, self.windowName, 0,
                           int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 2,
                           # -2 because we start from 0 and we show 2 frames at one time
                           self.trackbar)
        cv2.setMouseCallback(self.windowName, self.react)
        cv2.waitKey(0)

    def react(self, event, x, y, flags, params):
        """Mouse callback to set ROI

        event  - mouse event
        x: int  - x coordinate
        x: int  - y coordinate
        flags  - additional flags
        params - extra parameters
        """

        ids = []
        if self.horizontal:
            k = 1 if x > self.w else 0
            for obj in self.file1[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x - self.w * k < end[0] and start[1] < y < end[1]:
                    ids.append(obj['id'])
        else:
            k = 1 if y > self.h else 0
            for obj in self.file1[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x < end[0] and start[1] < y - self.h * k < end[1]:
                    ids.append(obj['id'])

        if event == cv2.EVENT_LBUTTONUP:
            for id in ids:
                newId = input('{} ->'.format(id))


        # elif event == cv2.EVENT_LBUTTONUP:
        #     if not self.drawingRect:
        #         return 0  # if person stops drawing clicking R button, but then releases L button
        #     flag = True
        #
        #     pTL = (min(self.drawingRect[0][0], self.drawingRect[1][0]),
        #            min(self.drawingRect[0][1], self.drawingRect[1][1]))
        #
        #     pBR = (max(self.drawingRect[0][0], self.drawingRect[1][0]),
        #            max(self.drawingRect[0][1], self.drawingRect[1][1]))
        #     self.drawingRect = None
        #     # Set the roiRect or reset it
        #     if pTL != pBR:
        #         # Adjust to X px padding
        #         pxBlock = 8
        #         dx = (pBR[0] - pTL[0]) % pxBlock
        #         if dx:
        #             dx = pxBlock - dx
        #         dy = (pBR[1] - pTL[1]) % pxBlock
        #         if dy:
        #             dy = pxBlock - dy
        #         self.roiRect.append((pTL, (pBR[0] + dx, pBR[1] + dy)))
        #     else:
        #         self.roiRect = self.roiRect[:-1]
        #
        # if event == cv2.EVENT_RBUTTONUP:
        #     """
        #     Delete drawn ROI by clicking inside it's area from the parent and child window
        #     """
        #     if self.drawingRect:
        #         self.drawingRect = None
        #     else:
        #         for i, ((x1, y1), (x2, y2)) in enumerate(self.roiRect):
        #             row = self.rowFrames[i]
        #             if x1 <= x <= x2 and y1 <= y <= y2 and int(
        #                     row.winfo_children()[-3].get()) <= self.trackerPos <= int(row.winfo_children()[-1].get()):
        #                 self.roiRect.remove(((x1, y1), (x2, y2)))
        #                 self.rowFrames.pop(i).destroy()

        if event == cv2.EVENT_MOUSEMOVE:
            self.drawRoi(ids)


    def drawRoi(self, ids=[]):
        """
        Draws the set ROI and calls tkWidget function to change info in the child box
        flag (bool): helping tool to understand if it is necessary to add drawn ROIs to the child box
        """
        img = [self.fframe.copy(), self.nframe.copy()]

        for i in [0, 1]:
            cv2.putText(img[i], str(self.trackerPos + i), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

            for obj in self.file1[self.trackerPos + i]['objects']:
                rt = max(1, int(self.w / self.w0) + 1)
                style = True if obj['id'] in ids else False
                img[i] = visualize_bbox(img[i], obj, thickness=rt, style=style)

        if self.horizontal:
            img = np.hstack(img)
        else:
            img = np.vstack(img)

        cv2.imshow(self.windowName, img)


    def trackbar(self, val):
        """
        If trackbar changes position the new frame is shown on the screen
        val (str): new trackbar position
        """
        self.trackerPos = val
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(val))
        _, self.fframe = self.vid.read()
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
    opt = parser.parse_args(
                            "-v /home/valia/AntVideos/Cflo_troph_count_7-02_7-09.mp4 -f1 "
                            "/home/valia/AntVideos/Cflo_troph_count_masked_6-00_6-31_MAL_withId.json "
                            "-vertical True".split())
    w, h = opt.wsize.split('x')
    flag = True if opt.horizontal else False
    App(opt.video, opt.filepath1, None, flag, int(w), int(h))
