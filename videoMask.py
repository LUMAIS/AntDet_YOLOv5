from re import findall
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import numpy as np
from colour import Color

def roi_processing(vidpath, color, rois, filename):
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
        if not findall(r'\d+,\d+,\d+,\d+', roi):
            print("Process interrupted \nInvalidArgument: wrong number of arguments in " + roi)
            return 0

        params = findall(r'\d+', roi)
        if len(params) < 4 or len(params) > 6:
            print("Process interrupted \nInvalidArgument: wrong number of arguments in " + roi)
            return 0

        x1, y1, w, h = list(map(int, params[:4]))
        if x1 > frame.shape[1] or y1 > frame.shape[0]:
            print("Process interrupted \nInvalidArgument: wrong coordinates of the roi specified as " + roi)
            return 0
        # get Left, Top, Width, Height and convert to int
        # separate xywh from frame intervals which should be processed
        w, h = min(w + x1, frame.shape[1]) - x1, min(h + y1, frame.shape[0]) - y1

        if len(params) == 6:
            numframes = list(map(int, params[4:]))
        else:
            if "^" in roi:
                numframes = list(map(int, params[4:])) + [total]
            elif "!" in roi:
                numframes = [1] + list(map(int, params[4:]))
            else:
                numframes = [1, total]

        # if shape is ellipse we need coordinates of a center and semiaxis' length
        center = (x1 + w/2, y1 + h/2)
        center = tuple(map(int, center))
        semiaxis = (int(w / 2), int(h / 2))
        x2, y2 = x1 + w, y1 + h

        for i in range(numframes[0] - 1, numframes[1]):
            mask = masks[i,]    #step back because counter starts wit the bias = 1
            if "ellipse" in roi:
                cv2.ellipse(mask, center, semiaxis, angle=0,
                            startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)
            else:
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    vid = cv2.VideoCapture(vidpath)
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                             vid.get(cv2.CAP_PROP_FPS), frame.shape[:2])

    bg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    bg[:,:] = Color(color).rgb[::-1]
    bg = bg * 255
    for mask in masks:
        _, frame = vid.read(cv2.IMREAD_UNCHANGED)
        if not np.any(mask):
            masked = frame
        elif color != "black":
            frame_mask = cv2.bitwise_and(frame, frame, mask=mask)
            bg_mask = cv2.bitwise_and(bg, bg, cv2.bitwise_not(mask))
            masked = cv2.bitwise_or(frame_mask, bg_mask)
            # cv2.addWeighted(frame_mask, 0.8, bg_mask, 1 - 0.9, 0.1, masked)
        else:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        writer.write(masked)
    vid.release()

    return 1

if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-v', '--vid', type=str, help='path to video')

    parser.add_argument('-c', '--color', type=str, default="black")
    parser.add_argument('-r', '--r', type=str, default=[], action='append',
                        help='LEFT,TOP,WIDTH,HEIGHT[;SHAPE=rect][^FRAME_START=1][!FRAME_FINISH=LAST_FRAME]]')
    parser.add_argument('-f', '--filename', type=str, help='name for a processed video')
    opt = parser.parse_args() #'-v E:\\100testimages.mp4 -c red -r 700,800,300,200;ellipse!20 -r 500,400,320,240^18!28 -f new.mp4'.split())

    roi_processing(*vars(opt).values())
