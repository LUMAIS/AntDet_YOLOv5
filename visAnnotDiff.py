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
    start = (int(tool['bbox']["left"]), int(tool['bbox']["top"]))
    end = (int(tool['bbox']["left"] + tool['bbox']["width"]),
           int(tool['bbox']["top"] + tool['bbox']["height"]))
    h = tool['color'].lstrip('#')
    color = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0)) #BGR
    if style:
        dashrect(image, start, end, color, thickness, style)
    else:
        cv2.rectangle(image, start, end, color, thickness)
    return image

# no need
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

def get_attr(obj: Dict) -> Dict:
    """
    Helps to track changes in attributes over time.

    Args:
        obj (dict): the piece of the labelbox annotation which describes only 1 object.
    Return:
        Shorten dictionary, where key is the attribute and value is a keyframe.
    """
    attrs = dict()
    if obj['classifications']:
        for classif in obj['classifications']:
            if classif['answers']:
                for answer in classif['answers']:
                    attrs[answer['value']] = answer['keyframe']
    return attrs



def main(annotated: str, reviewed: str, video: str, scale: float = 2, vidreview: str = None, keyframes: str = '1-$', epsilon: float = 0):
    """
    If video is given, draws annotation difference between given files.

    Args:
        annotated (str): path to annotation file before corrections
        reviewed (str): path to annotation file after corrections
        video (str): path to data with filename
        keyframes (str): intervals of frames that should be taken into account
    Return:
        numclschanges (int) - number of changes in total (among classes such as ant, ant-head, etc.)
        numattrchanges (int) - number of changes in total (among attributes such as blurry, side-view, etc.)
        numcorcls (int) - number of changes made by the reviewer (among classes)
        numcorattr (int) - number of changes made by the reviewer (among attributes)
    """
    with open(annotated) as f1:
        orFile = load(f1)
        f1.close()
    with open(reviewed) as f2:
        revFile = load(f2)
        f2.close()

    totalel = count_objects(orFile, keyframes, 0)
    total = len(orFile)
    print("------------Total-------------")
    print(f"Annotated classes: {totalel[0]}\nAnnotated attributes: {totalel[1]}".format(totalel))
    # orFile = shorten_file(orFile)
    # revFile = shorten_file(revFile)

    writer = None
    if video:
        vid = cv2.VideoCapture(video)
        if vidreview is not None:
            _, img = vid.read()
            height, width = img.shape[:2]
            writer = cv2.VideoWriter(vidreview, cv2.VideoWriter_fourcc(*'mp4v'),
                                     vid.get(cv2.CAP_PROP_FPS), (width, height))
    # changes in total made by both AI and in hand
    numclschanges = 0
    numattrchanges = 0
    # changes in made especially by the reviewer
    numcorcls = 0
    numcorattr = 0

    framelst = strparse(keyframes)

    for [beginning, ending] in framelst:
        ending = total if ending == '$' else ending

        if video:
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(beginning))
            _, img = vid.read()
        else:
            img = np.zeros((1, 1), np.uint8)

        for frameNum in range(int(beginning), int(ending) + 1):
            show = False
            rFrame = revFile[frameNum - 1]
            oFrame = orFile[frameNum - 1]

            for rObj in rFrame['objects']:
                flag = False # to track down if it's not a new object
                for oObj in oFrame['objects']:
                    if rObj['featureId'] == oObj['featureId']:
                        mistake = [abs(rObj['bbox'][dim] - oObj['bbox'][dim]) <= epsilon for dim in rObj['bbox'].keys()]
                        if mistake != [1] * 4:
                            numclschanges += 1
                            show = video
                            img = visualize_bbox(img, oObj)
                            img = visualize_bbox(img, rObj, style='dotted')
                            if rObj['keyframe']:
                                numcorcls += 1
                        flag = True
                        rAtr = get_attr(rObj)
                        oAtr = get_attr(oObj)
                        for attr, keyframe in rAtr.items():
                            if attr not in oAtr:
                                if keyframe:
                                    numcorattr += 1
                                numattrchanges += 1
                            # elif keyframe and not oAtr[attr]:
                            #     numcorattr += 1
                            #     numattrchanges += 1
                            # elif keyframe != oAtr[attr]:
                            #     numattrchanges += 1

                if not flag:
                    numcorcls += 1
                    numclschanges += 1
                    show = video
                    img = visualize_bbox(img, rObj, style='dashed')
            if writer is not None:
                writer.write(img)
            elif show:
                wTitle = 'frameNumber ' + str(frameNum)
                cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
                h, w = img.shape[:2]
                cv2.resizeWindow(wTitle, int(w / scale), int(h / scale))
                cv2.imshow(wTitle, img)
                key = 0
                while key != 32:  # space
                    key = cv2.waitKey(1) & 0xFF  # esc
                    if key == 27:
                        return 0
                cv2.destroyAllWindows()

            if video:
                _, img = vid.read()
                # frame = frame[:, :, ::-1]
    if writer is not None:
        writer.release()
    if video:
        vid.release()
    return numclschanges, numattrchanges, numcorcls, numcorattr


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
    opt = parser.parse_args()
    #'-a original_3-38_3-52.json -r review_3-38_3-52.json -k 1-35 -e 2 -v Cflo_troph_count_3-38_3-52.mp4 -o 1.mp4'.split())
    res = main(**vars(opt))
    print(
        (f"Corrected classes: {res[0]}\n"
         f"Corrected attributes: {res[1]}\n"
         f"Corrected classes in keyframes: {res[2]}\n"
         f"Corrected attributes in keyframes: {res[3]}").format(res))

