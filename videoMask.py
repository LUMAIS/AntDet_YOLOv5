#!AppData\Local\Programs\Python\Python39\python.exe
# -*- coding: utf-8 -*-
"""
:Description: Masking the trophallaxis events for ant videos.

:Authors: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2020-11-10
"""
import cv2
import numpy as np
import matplotlib._color_data as mcd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from re import split
from PIL import Image, ImageDraw
from numpy.random import randint
from random import choice, seed
from datetime import datetime



class Roi:
    """
    Created to process and keep relevant fields in it.

    Fields:
        xywh (tuple) - Top Left Width Height
        shape(str) - ellipse or rect
        start(int) - first frame to draw ROI on
        end (int, str) -  last frame to draw ROI on
    """

    def __init__(self, roi):
        self.xywh = None
        self.shape = None
        self.start = 1
        self.end = '$'
        self.__create(roi)

    def __create(self, roi):
        params = split(r',|;|\^|!', roi)
        self.xywh = tuple(map(int, params[:4]))
        self.shape = "ellipse" if "ellipse" in roi else "rect"
        params = split(r'ellipse|rect|\^|!', roi)[1:]
        params = [i for i in params if i != '']
        params = list(map(int, params))

        self.start = self.start if "^" not in roi else params[0]
        self.end = self.end if "!" not in roi else params[-1]


def roi_processing(vidpath: str, rois: str, filename: str, rand: bool = False, static: bool = True, color: str = "black"):
    """

    Args:
        vidpath (str): path to video data with filename
        rois (str): list of strings LEFT,TOP,WIDTH,HEIGHT [;SHAPE=rect][^FRAME_START=0][!FRAME_FINISH=LAST_FRAME]]
        filename (str): name for a new video
        rand (bool): True if background needs to be randomly colored
        color (str): color of a background written in English (red, blue, etc.)
    """
    vid = cv2.VideoCapture(vidpath)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in a video
    _, frame = vid.read()
    height, width = frame.shape[:2]
    print(width, height)
    now = datetime.now()
    seed(now.hour+now.minute+now.second)
    np.random.seed(now.hour+now.minute+now.second)

    roi_list = []
    for roi in rois:
        roi = Roi(roi)
        roi.end = total if roi.end == "$" else roi.end
        roi_list.append(roi)

    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                             vid.get(cv2.CAP_PROP_FPS), (width, height))

    # get hex code for a background color by name
    hcode = choice(list(mcd.CSS4_COLORS.values())) if not color else mcd.CSS4_COLORS[color]
    if static and rand:
        bg = Image.fromarray(randint(0, 256, (height, width, 3)).astype(np.uint8))
    else:
        bg = Image.new('RGB', (width, height), hcode)
    for i in range(1, total + 1):
        mask = Image.new("L", (width, height), 0)
        for roi in roi_list:
            if roi.start <= i and i <= roi.end:
                x1, y1, w, h = roi.xywh

                if x1 > width or y1 > height:
                    raise UnboundLocalError("Process interrupted \nInvalidArgument: wrong coordinates of the roi")

                # change coordinates of the ROI's Right Bottom corner if it is out of boundaries
                x2, y2 = min(w + x1, width), min(h + y1, height)

                draw = ImageDraw.Draw(mask)
                if roi.shape == "ellipse":
                    draw.ellipse((x1, y1, x2, y2), fill=255)
                else:
                    draw.rectangle((x1, y1, x2, y2), fill=255)
        if mask == Image.new("L", (width, height), 0):
            masked = frame
        else:
            im_frame = Image.fromarray(frame[:, :, ::-1])  # RGB
            if not static and rand:
                bg = Image.fromarray(randint(0, 256, (height, width, 3)).astype(np.uint8))
            elif not rand:
                hcode = choice(list(mcd.CSS4_COLORS.values()))
                bg = Image.new('RGB', (width, height), hcode)
            masked = Image.composite(im_frame, bg, mask)
            masked = np.array(masked)[:, :, ::-1]  # BGR
        writer.write(masked)
        _, frame = vid.read()
    writer.release()
    vid.release()


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-v', '--vidpath', type=str, help='path to video')

    parser.add_argument('-r', '--rois', type=str, default=[], action='append',
                        help='LEFT,TOP,WIDTH,HEIGHT[;SHAPE=rect][^FRAME_START=1][!FRAME_FINISH=LAST_FRAME]]')
    parser.add_argument('-f', '--filename', type=str, help='name for a processed video')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--color', type=str, help='color written as a word like pink, aqua, etc.')
    group.add_argument('-rand', action="store_true", help='True if background needs to be randomly colored')
    opt = parser.parse_args() #"-v E:\\work\\11-23_11-34.mp4 -r 737,378,125,159 -f 11-11.mp4".split())
    #1920x1061
    #^50(736, 411, 98, 164) to $
    # (793, 133, 164, 125) from 336 to 437
    #from 84 to 283
    # [(737, 378, 125, 159)]

    roi_processing(**vars(opt))
