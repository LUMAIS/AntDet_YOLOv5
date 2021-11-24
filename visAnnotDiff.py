#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: Evaluation and visualization script for annotation review.

:Authors: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2021-11-04
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from json import load
from typing import Dict, Any
from lbxTorch import count_objects, strparse
from collections import namedtuple

import cv2
import numpy as np


def dashline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        # s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def dashpoly(img, pts, color, thickness=1, style='dotted', ):
    # s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        dashline(img, s, e, color, thickness, style)


def dashrect(img: np.ndarray, pt1: tuple, pt2: tuple,
             color: tuple, thickness: int = 4, style: str ='dotted'):
    """
    Draws a dashed rectangle on an image

    Args:
        img (np.ndarray): image to draw a rectangle onto
        pt1 (tuple): coordinates of the left top corner
        pt2 (tuple): coordinates of the right bottom corner
        color (tuple): BGR color with values from 0 to 255
        style (str): dotted or else
    """
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    dashpoly(img, pts, color, thickness, style)


def visualize_bbox(image: np.ndarray, tool: Dict[str, Any], thickness: int = 2, style: str = '') -> np.ndarray:
    """
    Draws a bounding box on an image

    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
        style (str): False if rectangle without dashes have to be drawn
    Returns:
        image with a bounding box drawn on it.
    """
    start = (int(tool.bbox["left"]), int(tool.bbox["top"]))
    end = (int(tool.bbox["left"] + tool.bbox["width"]),
           int(tool.bbox["top"] + tool.bbox["height"]))
    h = tool.color.lstrip('#')
    color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0)) #BGR
    if style:
        dashrect(image, start, end, color, thickness, style)
    else:
        cv2.rectangle(image, start, end, color, thickness)
    return image


def shorten_file(jsFile: str) -> Dict[str, list]:
    """
    Gets only relevant information from annotation file

    Args:
        jsFile (list): json file which contains annotations
    Returns:
        {framenum: {<feature_id>: [<bbox>, <color>, <keyframe>]}} for jsfile
    """
    annotDict = dict()
    Annotation = namedtuple('Annotation', 'bbox color keyframe')

    for frame in jsFile:
        frameNum = frame['frameNumber']
        annotDict[frameNum] = dict()
        for obj in frame['objects']:
            feature_id = obj['featureId']
            annotDict[frameNum][feature_id] = Annotation(obj['bbox'], obj['color'], obj['keyframe'])
    return annotDict


def main(annotated: str, reviewed: str, video: str, scale: float = 2, vidreview: str = None, keyframes: str = '1-$', epsilon: float = 0):
    """
    If video is given, draws annotation difference between given files.

    Args:
        annotated (str): path to annotation file before corrections
        reviewed (str): path to annotation file after corrections
        video (str): path to data with filename
        keyframes (str): intervals of frames that should be taken into account
    """
    with open(annotated) as f1:
        orFile = load(f1)
        f1.close()
    with open(reviewed) as f2:
        revFile = load(f2)
        f2.close()

    totalel = count_objects(orFile, keyframes, 0)
    total = len(orFile)
    print("Number of annotated objects is", totalel)
    orFile = shorten_file(orFile)
    revFile = shorten_file(revFile)

    writer = None
    if video:
        vid = cv2.VideoCapture(video)
        if vidreview is not None:
            _, frame = vid.read()
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(vidreview, cv2.VideoWriter_fourcc(*'mp4v'),
                                     vid.get(cv2.CAP_PROP_FPS), (width, height))

    numchanges = 0
    numcorpos = 0
    framelst = strparse(keyframes)

    for [beginning, ending] in framelst:
        ending = total if ending == '$' else ending

        if video:
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(beginning))
            _, frame = vid.read()
        else:
            frame = np.zeros((1, 1), np.uint8)

        for frameNum in range(int(beginning), int(ending) + 1):
            show = False
            for feature_id, revobject in revFile[frameNum].items():
                try:
                    orBbox = orFile[frameNum][feature_id].bbox
                    mistake = [abs(orBbox[key] - revobject.bbox[key]) <= epsilon for key in revobject.bbox]
                    if mistake != [1] * 4:
                        numchanges += 1
                        show = video
                        frame = visualize_bbox(frame, orFile[frameNum][feature_id])
                        frame = visualize_bbox(frame, revFile[frameNum][feature_id], style='dotted')
                        if revobject.keyframe:
                            numcorpos += 1
                except KeyError:
                    numcorpos += 1
                    numchanges += 1
                    show = video
                    frame = visualize_bbox(frame, revobject, style='dashed')
            if writer is not None:
                writer.write(frame)
            elif show:
                wTitle = 'frameNumber ' + str(frameNum)
                cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
                h, w = frame.shape[:2]
                cv2.resizeWindow(wTitle, int(w / scale), int(h / scale))
                cv2.imshow(wTitle, frame)
                key = 0
                while key != 32:  # space
                    key = cv2.waitKey(1) & 0xFF  # esc
                    if key == 27:
                        return 0
                cv2.destroyAllWindows()

            if video:
                _, frame = vid.read()
                # frame = frame[:, :, ::-1]
    if writer is not None:
        writer.release()
    if video:
        vid.release()
    return numchanges, numcorpos


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-a', '--annotated', type=str, help='Path to the JSON file of original annotations')
    parser.add_argument('-r', '--reviewed', type=str, help='Path to the JSON file of reviewed annotations')

    parser.add_argument('-v', '--video', type=str, default ='', help='Path to the original video')
    parser.add_argument('-o', '--output-video', dest='vidreview', type=str, help='Output video instead of the interactive analysis')
    parser.add_argument('-k', '--keyframes', type=str, default='1-$', help='Target intervals of frames if necessary')
    parser.add_argument('-e', '--epsilon', type=float, default=0, help='The maximum permissible error of the bbox dimension')
    opt = parser.parse_args('-a E:\\work\\original_3-38_3-52.json -r E:\\work\\test_rev.json --keyframes 1-$ -e 1'.split())
    # #'-v E:\\work\\3-38_3-52.mp4 -a E:\\work\\original_3-38_3-52.json -r E:\\work\\review_ind.json --keyframes 1-6 -o out.mp4'.split())
    #Number of annotated objects is 12858
    # Number of corrected positions in fact is 2734     22136 < 22885 < 24202
    # Number of corrections indeed is 58

    #12858 / 45360
    res = main(**vars(opt))
    print("Number of corrected positions in fact is", res[0])
    print("Number of corrections indeed is", res[1])

