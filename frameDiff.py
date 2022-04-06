#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: GUI for validation and editing of automatic adjustment of object ID on successive frames.

:Author: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2022-03-25
"""
import cv2
import os
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
    h = tool['color'].lstrip('#')
    color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))  # BGR

    k = 1 if not bold else 2
    if not dashed:
        cv2.rectangle(image, start, end, color, thickness*k)
    else:
        dashrect(image, start, end, color, thickness, 'dashed')

    if tool['id'][:2] != "ah" and tool['id'][0] == "a":
        cv2.putText(image, tool['id'], (start[0], end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1*k)
    else:
        cv2.putText(image, tool['id'], (start[0], start[1] - 3),
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
        self.notClear = []
        self.mode = '1'
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

        k = 2 if self.horizontal else 1/2

        # dummy resizing in context of smart scaling. But is still dummy
        if self.h / h0 > k * self.w / w0:
            cv2.resizeWindow(self.windowName, int(k * self.w * h0 / self.h), h0)
        else:
            cv2.resizeWindow(self.windowName, w0, int(self.h * w0 / self.w / k))

        self.drawRoi()
        cv2.createTrackbar(self.trTitle, self.windowName, 0,
                           int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 2,
                           # -2 because we start from 0 and we show 2 frames at one time
                           self.trackbar)
        cv2.setMouseCallback(self.windowName, self.react)
        print('-- To choose the object and to change it featureId just click inside it \n'
              '-- If you want to change the featureId of the nested object click inside it either \n'
              '-- Input new number to change the featureId of current object.  \n'
              '-- To cancel featureId changing press Enter \n'
              '-- To switch to the previous frame press P \n'
              '-- To switch to the next frame press N \n'
              '-- To switch the mode of linking lines press 1-3\n')
        while(1):
            key = cv2.waitKey(1)
            # Quit: escape or q
            if key in (27, ord('q')):
                with open('improved.json', 'w') as f:
                    json.dump(self.file, f)
                cv2.destroyAllWindows()
                break
            elif key == ord('n') and self.trackerPos < int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 3:
                self.trackbar(self.trackerPos + 1)
            elif key == ord('p') and self.trackerPos > 0:
                self.trackbar(self.trackerPos - 1)
            elif key in (ord('1'), ord('2'), ord('3')):
                self.mode = chr(key)
                print('You have switched the line drawing mode to', self.mode)


    def react(self, event, x, y, flags, params):
        """Mouse callback to choose ROIs to correct their Id's

        event  - mouse event
        x: int  - x coordinate
        x: int  - y coordinate
        flags  - additional flags
        params - extra parameters
        """

        ids = []
        if self.horizontal:
            k = 1 if x > self.w else 0
            for obj in self.file[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x - self.w * k < end[0] and start[1] < y < end[1]:
                    ids.append(obj['id'])
        else:
            k = 1 if y > self.h else 0
            for obj in self.file[self.trackerPos + k]['objects']:
                start = (int(obj['bbox']["left"]), int(obj['bbox']["top"]))
                end = (int(obj['bbox']["left"] + obj['bbox']["width"]),
                       int(obj['bbox']["top"] + obj['bbox']["height"]))
                if start[0] < x < end[0] and start[1] < y - self.h * k < end[1]:
                    ids.append(obj['id'])

        if event == cv2.EVENT_LBUTTONUP:
            for id in ids:
                letter = findall(r'\D+', id)[0]
                newId = input('{} -> {}'.format(id, letter))
                try:
                    newId = int(newId)
                    newId = letter + str(newId)
                    for j in range(self.trackerPos + 1, int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))):
                        changed = {'old': -1,
                                   'new': -1}
                        for i, obj in enumerate(self.file[self.trackerPos + 1]['objects']):
                            if obj['id'] == id:
                                changed['old'] = i
                            elif obj['id'] == newId:
                                changed['new'] = i
                        if changed['old'] != -1:
                            if changed['new'] != -1:
                                self.file[j]['objects'][changed['old']]['id'],\
                                self.file[j]['objects'][changed['new']]['id'] = newId, id
                            else:
                                self.file[j]['objects'][changed['old']]['id'] = newId

                except ValueError: pass

        if event == cv2.EVENT_MOUSEMOVE:
            self.drawRoi(ids)


    def drawRoi(self, ids=[]):
        """
        Draws the set ROIs with their Id's
        Args:
            ids (list): carries Id's which should be highlighted with bold boundaries
        """
        img = [self.fframe.copy(), self.nframe.copy()]
        rt = max(1, int(self.w / self.w0) + 1)

        for i in [0, 1]:
            cv2.putText(img[i], str(self.trackerPos + i), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

            for obj in self.file[self.trackerPos + i]['objects']:
                bold = True if obj['id'] in ids else False
                img[i] = visualize_bbox(img[i], obj, thickness=rt, bold=bold)

        if self.horizontal:
            img = np.hstack(img)
        else:
            img = np.vstack(img)
        if self.mode in ('2', '1'):
            for p_obj in self.file[self.trackerPos]['objects']:
                for n_obj in self.file[self.trackerPos + 1]['objects']:
                    if p_obj['id'] == n_obj['id']:
                        bold = True if p_obj['id'] in ids else False
                        # change after new script is ready
                        dashed = True if p_obj['id'] in self.notClear else False
                        if self.mode == '2' and bold:
                            img = self.visualize_line(img, p_obj, n_obj, thickness=rt, bold=bold)
                            cv2.imshow(self.windowName, img)
                            return 0
                        elif self.mode == '1':
                            img = self.visualize_line(img, p_obj, n_obj, thickness=rt, bold=bold)

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

        h = p_obj['color'].lstrip('#')
        color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))  # BGR

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
    parser.add_argument('-vid', '--video', type=str, help='path to video', required=True)
    parser.add_argument('-a', '--annotations', type=str, help='path to the MAL annotations', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-hor', '--horizontal', type=bool, help="type of images' stack")
    group.add_argument('-ver', '--vertical', type=bool, help="type of images' stack")

    parser.add_argument('-wsize', type=str, default="1600x1200", help='Your screen parameters WxH')
    opt = parser.parse_args()
    w, h = opt.wsize.split('x')
    flag = True if opt.horizontal else False
    flag = True if not opt.horizontal and not opt.vertical else flag
    App(opt.video, opt.annotations, flag, int(w), int(h))
