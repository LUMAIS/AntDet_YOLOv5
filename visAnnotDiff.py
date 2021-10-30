from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import numpy as np
from json import load
# import math
from typing import Dict, Any
from IPython.display import clear_output
from matplotlib import pyplot as plt

#code taken from StackOverflow
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
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def dashpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        dashline(img, s, e, color, thickness, style)


def dashrect(img, pt1, pt2, color, thickness=4, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    dashpoly(img, pts, color, thickness, style)


def visualize_bbox(image: np.ndarray, tool: Dict[str, Any], dash: bool = False) -> np.ndarray:
    """
    Draws a bounding box on an image

    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
        dash (bool): False if rectangle without dashes have to be drawn
    Returns:
        image with a bounding box drawn on it.
    """
    start = (int(tool[0]["left"]), int(tool[0]["top"]))
    end = (int(tool[0]["left"] + tool[0]["width"]),
           int(tool[0]["top"] + tool[0]["height"]))
    h = tool[1].lstrip('#')
    color = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if dash:
        dashrect(image, start, end, color)#, style='dashed')
    else:
        cv2.rectangle(image, start, end, color, 4)
    return image

def shorten_file(jsFile: str) -> Dict[str, list]:
    """
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


def main(file1: str, file2: str, vidpath: str, fframe: int = 0):
    """

    Args:
        file1 (str): path to annotation file before corrections
        file2 (str): path to annotation file after corrections
        vidpath (str): path to data with filename
        fframe (int): last changed frame (if needed)
    """
    with open(file1) as f1:
        orFile = load(f1)
        f1.close()
    with open(file2) as f2:
        revFile = load(f2)
        f2.close()

    orFile = shorten_file(orFile)
    revFile = shorten_file(revFile)

    vid = cv2.VideoCapture(vidpath)
    total = fframe if fframe else int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame = vid.read()
    #frame = frame[:, :, ::-1]
    for frameNum in range(1, total + 1):
        for feature_id in revFile[frameNum]:
            try:
                if orFile[frameNum][feature_id] != revFile[frameNum][feature_id]:
                    frame = visualize_bbox(frame.astype(np.uint8), orFile[frameNum][feature_id])
                    frame = visualize_bbox(frame.astype(np.uint8), revFile[frameNum][feature_id], dash=True)
            except KeyError:
                frame = visualize_bbox(frame.astype(np.uint8), revFile[frameNum][feature_id], dash=True)
        plt.figure(1)
        plt.imshow(frame)
        plt.title('frameNumber ' + str(frameNum + 1))
        plt.pause(0.4)
        plt.clf()

        success, frame = vid.read()
        if success:
            clear_output(wait=True)
        #frame = frame[:, :, ::-1]
    vid.release()


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-orf', '--file1', type=str, help='path to a non corrected jsfile')
    parser.add_argument('-ref', '--file2', type=str, help='path to a reviewed corrected jsfile')

    parser.add_argument('-vid', '--v', type=str, help='path to a video')
    parser.add_argument('-fframe', '--fframe', type=int, default=0, help='Final annotated frame')
    opt = parser.parse_args()#'-vid E:\\100testimages.mp4 -orf E:\\original.json -ref E:\\reviewed.json -fframe 9'.split())

    main(*vars(opt).values())