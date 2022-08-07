#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: GUI for validation and editing of automatic adjustment of object ID on successive frames.

:Author: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2022-03-25
"""
import os.path

import cv2
import json
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from visAnnotDiff import dashrect
from re import findall


def visualize_bbox(image: np.ndarray, tool, bold=False, dashed=False, thickness: int = 2) -> np.ndarray:
    """
    Draws a bounding box on an image

    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
        bold (str): False if rectangle should be without bold boundaries
    Returns:
        image with a bounding box drawn on it.
    """
    start = (int(tool['bbox']["left"]), int(tool['bbox']["top"]))
    end = (int(tool['bbox']["left"] + tool['bbox']["width"]),
           int(tool['bbox']["top"] + tool['bbox']["height"]))

    if 'color' in tool.keys():
        h = tool['color'].lstrip('#')
        color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))  # BGR
    else:
        color = (0, 0, 1)

    k = 1 if not bold else 2
    if not dashed:
        cv2.rectangle(image, start, end, color, thickness * k)
    else:
        dashrect(image, start, end, color, thickness * k, 'dashed')

    if tool['value'] == "ant":  # tool['id'][:2] != "ah" and tool['id'][0] == "a":
        cv2.putText(image, tool['featureId'], (start[0], end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1 * k)
    else:
        cv2.putText(image, tool['featureId'], (start[0], start[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1 * k)

    return image


class App:
    def __init__(self, video, filepath, horizontal=True, w0=1880, h0=1021):

        with open(filepath, 'r') as f:
            self.file = json.load(f)

        self.horizontal = horizontal
        self.w0 = w0
        self.vidpath = video
        self.vid = cv2.VideoCapture(video)
        # left (first frame) and right frame (next frame) respectively
        _, self.fframe = self.vid.read()
        _, self.nframe = self.vid.read()

        # self.notClear = {'2': ['a2', 'ah5']}    # will be changed after another script will become completed
        self.notClear = {}
        self.linemode = '1'
        self.h, self.w = self.fframe.shape[:2]
        self.windowName = 'image'
        self.trTitle = 'tracker'
        self.trackerPos = 0

        self.lastevent = None
        self.tochange = None
        self.roi = None  # coordinates of the ROI
        self.pic = None  # coordinates of the ROI
        self.draw = False  # if True then GUI reacts to mouse movements
        self.roimode = False  # if True then ROI can be drawn via mouse
        self.mask = np.zeros((self.h + self.h * (not self.horizontal),
                              self.w + self.w * self.horizontal, 3), dtype=np.uint8)

        self.begin(filepath, w0, h0)

    def begin(self, filepath, w0, h0):
        """
        Starts
        + frame showing
        + creates a trackbar for navigation
        + set mouse callback to draw Roi
        + calls drawROI
        """
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

        k = 2 if self.horizontal else 1 / 2

        # dummy resizing in context of smart scaling. But is still dummy
        if self.h / h0 > k * self.w / w0:
            cv2.resizeWindow(self.windowName, int(k * self.w * h0 / self.h), h0)
        else:
            cv2.resizeWindow(self.windowName, w0, int(self.h * w0 / self.w / k))

        self.drawRoi()
        cv2.createTrackbar(self.trTitle, self.windowName, 0,
                           int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 2,
                           # -2 because we start from 0, and we show 2 frames at one time
                           self.trackbar)
        cv2.setMouseCallback(self.windowName, self.react)
        print('-- To choose the object and to change it featureId just click inside it \n'
              '-- If you want to change the featureId of the nested object click inside it either \n'
              '-- Input new number to change the featureId of current object.  \n'
              '-- To cancel featureId changing press Enter \n'
              '-- To switch to the previous frame press P \n'
              '-- To switch to the next frame press N \n'
              '-- To switch the mode of linking lines press 1-3\n'
              '-- To remove the bbox press Del \n')
        while 1:
            key = cv2.waitKey(1)
            # Quit: escape or q

            if key in (27, ord('q')):
                filename = filepath.rstrip('.json') + '_imp.json'
                if os.path.exists(filename):
                    i = 1
                    while os.path.exists(
                            filepath.rstrip('.json') + '_imp_{}.json'.format(i)):
                        i += 1
                    filename = filepath.rstrip('.json') + '_imp_{}.json'.format(i)

                with open(filename, 'w') as f:
                    json.dump(self.file, f)
                cv2.destroyAllWindows()
                break
            elif key == 255:  # Del
                self.react(x=0, y=0, event=cv2.EVENT_RBUTTONUP)
            elif key == ord('n') and self.trackerPos < int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 3:
                self.trackbar(self.trackerPos + 1)
            elif key == ord('p') and self.trackerPos > 0:
                self.trackbar(self.trackerPos - 1)
            elif key in (ord('1'), ord('2'), ord('3')):
                self.linemode = chr(key)
                print('You have switched the line drawing mode to', self.linemode)
            elif key == ord('d'):
                self.roimode = True
                print('You have switched the to drawing ROI mode')

    def react(self, event, x, y, flags=None, params=None):
        """Mouse callback to choose ROIs to correct their Id's

        event  - mouse event
        x: int  - x coordinate
        x: int  - y coordinate
        flags  - additional flags
        params - extra parameters
        """

        ids = []
        onmouse = ''

        if self.horizontal:
            k = 1 if x > self.w else 0
            for obj in self.file[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x - self.w * k < end[0] and start[1] < y < end[1]:
                    s = (np.array(start) - np.array(end))[0] * (np.array(start) - np.array(end))[1]
                    ids.append([obj['featureId'], s])
        else:
            k = 1 if y > self.h else 0
            for obj in self.file[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x < end[0] and start[1] < y - self.h * k < end[1]:
                    s = (np.array(start) - np.array(end))[0] * (np.array(start) - np.array(end))[1]
                    ids.append([obj['featureId'], s])
        if ids:
            def findmin(l):
                res = None if not l else l[0]
                for pair in l:
                    if res[1] > pair[1]:
                        res = pair.copy()
                return res[0]

            onmouse = findmin(ids)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            print('hello')

        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self.roi and self.roimode:
                self.roi = [(x, y), (x, y)]
                self.pic = [(x, y), (x, y)]
                self.draw = True


        elif event == cv2.EVENT_LBUTTONUP and not self.draw:
            if not onmouse:
                if (self.horizontal and x > self.w) or (not self.horizontal and y > self.h):
                    self.tochange = None
            else:
                if self.tochange and ((self.horizontal and x < self.w) or (not self.horizontal and y < self.h)):
                    for j in range(self.trackerPos + 1, int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))):
                        indexSwitch = {'old': -1,
                                       'new': -1}
                        for i, obj in enumerate(self.file[j]['objects']):
                            if obj['featureId'] == self.tochange:
                                indexSwitch['old'] = i
                            elif obj['featureId'] == onmouse:
                                indexSwitch['new'] = i
                        if indexSwitch['old'] != -1:
                            if indexSwitch['new'] != -1:
                                self.file[j]['objects'][indexSwitch['old']]['featureId'], \
                                self.file[j]['objects'][indexSwitch['new']]['featureId'] = onmouse, self.tochange
                            else:
                                self.file[j]['objects'][indexSwitch['old']]['featureId'] = self.tochange
                    # self.tochange = onmouse
                else:
                    self.tochange = onmouse

        elif (event == cv2.EVENT_LBUTTONUP and self.roimode) or (event == cv2.EVENT_MOUSEMOVE and self.draw):

            if event == cv2.EVENT_LBUTTONUP:
                self.draw = False
                self.roimode = False
                print('Drawing ROI mode finished')
            else:
                self.roi[1] = (x, y)

            pTL = (min(self.roi[0][0], self.roi[1][0]),
                   min(self.roi[0][1], self.roi[1][1]))
            pBR = (max(self.roi[0][0], self.roi[1][0]),
                   max(self.roi[0][1], self.roi[1][1]))

            if pTL != pBR:
                # Adjust to X px padding
                pxBlock = 8
                dx = (pBR[0] - pTL[0]) % pxBlock
                if dx:
                    dx = pxBlock - dx
                dy = (pBR[1] - pTL[1]) % pxBlock
                if dy:
                    dy = pxBlock - dy
                self.pic = [pTL, (pBR[0] + dx, pBR[1] + dy)]

            # print('\nROI size is corrected by ({}, {}): ({}, {})'.format(dx, dy, roiRect[1][0]-roiRect[0][0], roiRect[1][1]-roiRect[0][1]))

        if event == cv2.EVENT_RBUTTONUP:
            self.roi = None
            self.pic = None
            self.draw = False
            self.mask *= 0

        self.lastevent = event
        self.drawRoi(onmouse)

    def drawRoi(self, highlight=''):
        """
        Draws the set ROIs with their ids.
        Args:
            highlight (string): carries ids, which should be highlighted with bold boundaries
        """
        img = [self.fframe.copy(), self.nframe.copy()]
        rt = max(1, int(self.w / self.w0) + 1)
        x1, y1 = (0, 0)
        x2, y2 = (self.w, self.h)

        # ----------------------- bboxes --------------------------
        for i in [0, 1]:
            cv2.putText(img[i], str(self.trackerPos + i), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

            for obj in self.file[self.trackerPos + i]['objects']:
                bold = True if obj['featureId'] == highlight else False
                dashed = True if obj['featureId'] == self.tochange and i == 1 else False
                # if i == 1:
                #     try:
                #         dashed = True if obj['featureId'] in self.notClear[str(self.trackerPos + 2)] else False
                #     except KeyError:
                #         pass
                img[i] = visualize_bbox(img[i], obj, thickness=rt, bold=bold, dashed=dashed)

        if self.horizontal:
            img = np.hstack(img)
        else:
            img = np.vstack(img)

        # ----------------------- Mask due to ROI --------------------------
        if self.roi and self.roi[0] != self.roi[1]:
            k = int(self.horizontal)
            if self.pic[0][0] < self.w and self.pic[0][1] < self.h:
                x1, y1 = self.roi[0]
                x2, y2 = (min(self.pic[1][0], self.w), min(self.pic[1][1], self.h))
            else:
                x1, y1 = (self.pic[0][0] - self.w * k, self.pic[0][1] - self.h * (not k))
                x2, y2 = (min(self.pic[1][0] - self.w * k, self.w),
                          min(self.pic[1][1] - self.h * (not k), self.h))

            self.mask = np.zeros(img.shape, dtype=np.uint8)
            self.mask = cv2.rectangle(self.mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            self.mask = cv2.rectangle(self.mask, (x1 + self.w * k, y1 + self.h * (not k)),
                                      (x2 + self.w * k, y2 + self.h * (not k)), (255, 255, 255), -1)

            img = cv2.bitwise_and(img, self.mask)

        # ----------------------- lines due to features --------------------------
        if self.linemode in ('2', '1'):
            # obj_in_area = 0
            for p_obj in self.file[self.trackerPos]['objects']:
                mid = (int(p_obj['bbox']["left"] + p_obj['bbox']["width"] / 2),
                       int(p_obj['bbox']["top"] + p_obj['bbox']["height"] / 2))

                if x1 < mid[0] < x2 and y1 < mid[1] < y2:
                    for n_obj in self.file[self.trackerPos + 1]['objects']:
                        if p_obj['featureId'] == n_obj['featureId']:
                            bold = True if p_obj['featureId'] == highlight else False
                            if (self.linemode == '2' and bold) or self.linemode == '1':
                                # obj_in_area += 1 if self.mode == '2' and bold else 0
                                img = self.visualize_line(img, p_obj, n_obj, thickness=rt, bold=bold)
                                # if obj_in_area == 2:
                                #     cv2.imshow(self.windowName, img)
                                #     return 0

        cv2.imshow(self.windowName, img)

    def visualize_line(self, image: np.ndarray, p_obj, n_obj, bold=False, thickness: int = 2) -> np.ndarray:
        """
        Draws a bounding box on an image

        Args:
            image (np.ndarray): image to draw a bounding box onto
            tool (Dict[str,any]): Dict response from the export
            bold (str): False if rectangle should be without bold boundaries
        Returns:
            image with a bounding box drawn on it.
        """

        start = (int(p_obj['bbox']["left"] + p_obj['bbox']["width"] / 2),
                 int(p_obj['bbox']["top"] + p_obj['bbox']["height"] / 2))
        if self.horizontal:
            end = (int(n_obj['bbox']["left"] + n_obj['bbox']["width"] / 2 + self.w),
                   int(n_obj['bbox']["top"] + n_obj['bbox']["height"] / 2))
        else:
            end = (int(n_obj['bbox']["left"] + n_obj['bbox']["width"] / 2),
                   int(n_obj['bbox']["top"] + n_obj['bbox']["height"] / 2 + self.h))

        if 'color' in p_obj.keys():
            h = p_obj['color'].lstrip('#')
            color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))  # BGR
        else:
            color = (0, 0, 1)

        k = 1 if not bold else 2
        cv2.line(image, start, end, color=color, thickness=thickness * k)

        return image

    def trackbar(self, val):
        """
        If trackbar changes position the new frame is shown on the screen
        val (str): new trackbar position
        """
        self.trackerPos = val
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(val))
        _, self.fframe = self.vid.read()
        _, self.nframe = self.vid.read()
        self.drawRoi()


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-vid', '--video', default='test-parameters/Cflo_troph_count_masked_6-00_6-31.mp4',
                        type=str, help='path to video')
    parser.add_argument('-a', '--annotations',
                        default='test-parameters/Cflo_troph_count_masked_6-00_6-31_MAL_withId.json',
                        type=str, help='path to the MAL annotations')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-hor', '--horizontal', action="store_true",help="type of images' stack")
    group.add_argument('-ver', '--vertical', action="store_true", help="type of images' stack")

    parser.add_argument('-wsize', type=str, default="1600x1200", help='Your screen parameters WxH')

    print()
    # opt = parser.parse_args()
    # using test-parameters
    # opt = parser.parse_args("-vid /home/valia/AntVideos/Cflo_troph_count_3-38_3-52.mp4 "
    #                         "-a /home/valia/Downloads/lo.json -ver".split())
    # "-a /home/valia/AntVideos/Cflo_troph_count_3-38_3-52_withId.json".split())

    w, h = opt.wsize.split('x')
    flag = True if opt.horizontal else False
    flag = True if not opt.horizontal and not opt.vertical else flag
    App(opt.video, opt.annotations, flag, int(w), int(h))
