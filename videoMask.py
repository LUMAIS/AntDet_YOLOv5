from re import findall
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import numpy as np


def roi_processing(vidpath, rois, filename):
    """

    :param vidpath: path to data with filename
    :param rois: list of strings LEFT,TOP,WIDTH,HEIGHT [;SHAPE=rect][^FRAME_START=0][!FRAME_FINISH=LAST_FRAME]]
    :param filename: name for a new video
    :return:
    """

    vid = cv2.VideoCapture(vidpath)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #number of frames in a video
    _, frame = vid.read()

    # ndarray of masks for each frame
    masks = np.zeros((total, frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for roi in rois:
        xywh = list(map(int, findall(r'\d+', roi)[:4]))        #get Left, Top, Width, Height and convert to int
        numframes = list(map(int, roi.split('^')[-1].split("!"))) #frame intervals which should be processed

        # if shape is ellipse we need coordinates of a center and semiaxis' length
        center = (xywh[0] + xywh[2]/2, xywh[1] + xywh[3]/2)
        center = tuple(map(int, center))
        semiaxis = (int(xywh[2] / 2), int(xywh[3] / 2))

        for i in range(numframes[0], numframes[1] + 1):
            mask = masks[i - 1,] #step back because counter starts wit the bias = 1
            if "ellipse" in roi:
                cv2.ellipse(mask, center, semiaxis, angle=0,
                            startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)
            else:
                x2, y2 = xywh[0] + xywh[2], xywh[1] + xywh[3]
                cv2.rectangle(mask, xywh[:2], (x2, y2), (255, 255, 255), -1)

    vid = cv2.VideoCapture(vidpath)
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                             vid.get(cv2.CAP_PROP_FPS), frame.shape[:2])
    for mask in masks:
        _, frame = vid.read()
        masked = frame if not np.any(mask) else cv2.bitwise_and(frame, frame, mask=mask)
        writer.write(masked)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-v', '--vid', type=str, help='path to video')

    parser.add_argument('-r', '--r', type=str, default=[], action='append',
                        help='LEFT,TOP,WIDTH,HEIGHT[;SHAPE=rect][^FRAME_START=0][!FRAME_FINISH=LAST_FRAME]]')
    parser.add_argument('-f', '--filename', type=str, help='name for a processed video')
    opt = parser.parse_args() #'-v E:\\100testimages.mp4 -r 700,800,300,200;ellipse^10!20 -r 500,400,320,240;^18!28 -f new.mp4'.split())

    roi_processing(*vars(opt).values())