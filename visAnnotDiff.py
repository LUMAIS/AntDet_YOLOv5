from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from json import load
from typing import Dict, Any
from lbxTorch import count_objects, strparse

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
    start = (int(tool[0]["left"]), int(tool[0]["top"]))
    end = (int(tool[0]["left"] + tool[0]["width"]),
           int(tool[0]["top"] + tool[0]["height"]))
    h = tool[1].lstrip('#')
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
        {framenum: {<feature_id>: [<bbox>, <color>]}} for jsfile
    """
    annotDict = dict()

    for frame in jsFile:
        frameNum = frame['frameNumber']
        annotDict[frameNum] = dict()
        for obj in frame['objects']:
            feature_id = obj['featureId']
            annotDict[frameNum][feature_id] = [obj['bbox'], obj['color']]
    return annotDict


def main(file1: str, file2: str, vidpath: str, w0: float = 2, keyframes: str = '1-$') -> int:
    """
    If vidpath is given, draws annotation difference between given files.

    Args:
        file1 (str): path to annotation file before corrections
        file2 (str): path to annotation file after corrections
        vidpath (str): path to data with filename
        w0 (float): demonstration scale coefficient
        keyframes (str): intervals of frames that should be taken into account
    """
    with open(file1) as f1:
        orFile = load(f1)
        f1.close()
    with open(file2) as f2:
        revFile = load(f2)
        f2.close()

    totalel = count_objects(orFile, keyframes, 0)
    total = len(orFile)
    print("Total number of annotated objects is ", totalel)
    orFile = shorten_file(orFile)
    revFile = shorten_file(revFile)

    if vidpath:
        vid = cv2.VideoCapture(vidpath)

    totcorected = 0
    framelst = strparse(keyframes)

    for [beginning, ending] in framelst:
        ending = total if ending == '$' else ending

        if vidpath:
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(beginning))
            _, frame = vid.read()
        else:
            frame = np.zeros((1, 1), np.uint8)

        for frameNum in range(int(beginning), int(ending) + 1):
            show = False
            for feature_id in revFile[frameNum]:
                try:
                    if orFile[frameNum][feature_id] != revFile[frameNum][feature_id]:
                        show = vidpath
                        totcorected += 1
                        frame = visualize_bbox(frame, orFile[frameNum][feature_id])
                        frame = visualize_bbox(frame, revFile[frameNum][feature_id], style='dotted')
                except KeyError:
                    totcorected += 1
                    show = vidpath
                    frame = visualize_bbox(frame.astype(np.uint8), revFile[frameNum][feature_id], style='dashed')
            if show:
                wTitle = 'frameNumber ' + str(frameNum)
                cv2.namedWindow(wTitle, cv2.WINDOW_NORMAL)
                h, w = frame.shape[:2]
                cv2.resizeWindow(wTitle, int(w / w0), int(h / w0))
                cv2.imshow(wTitle, frame)
                key = 0
                while key != 32:  # space
                    key = cv2.waitKey(1) & 0xFF  # esc
                    if key == 27:
                        return 0
                cv2.destroyAllWindows()

            if vidpath:
                _, frame = vid.read()
                # frame = frame[:, :, ::-1]
    if vidpath:
        vid.release()
    return totcorected


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-orf', '--file1', type=str, help='path to a non corrected jsfile')
    parser.add_argument('-ref', '--file2', type=str, help='path to a reviewed corrected jsfile')

    parser.add_argument('-vid', '--vidpath', type=str, default ='', help='path to a video')
    parser.add_argument('-fframe', '--keyframes', type=str, help='intervals of frames that should be taken into account')
    opt = parser.parse_args() #'-orf E:\\work\\original_3-38_3-52.json -ref E:\\work\\review_ind.json --keyframes 1-$'.split())
    #-vid E:\\work\\3-38_3-52.mp4

    print("Total number of corrected elements is ", main(**vars(opt)))
