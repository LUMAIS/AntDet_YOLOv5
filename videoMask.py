from re import findall
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import numpy as np
from colour import Color
from PIL import Image, ImageDraw

def roi_processing(vidpath, color, rois, filename):
    """

    Args:
        vidpath (str): path to data with filename
        color (str): color of a background written in English (red, blue, etc.)
        rois (str): list of strings LEFT,TOP,WIDTH,HEIGHT [;SHAPE=rect][^FRAME_START=0][!FRAME_FINISH=LAST_FRAME]]
        filename (str): name for a new video
    Returns:
        0 if process ended with errors
    """
    vid = cv2.VideoCapture(vidpath)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #number of frames in a video
    _, frame = vid.read()

    # ndarray of masks for each frame
    masks = [Image.new("L", frame.shape[:2], 0) for i in range(total)]

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
        x2, y2 = min(w + x1, frame.shape[1]), min(h + y1, frame.shape[0])

        if len(params) == 6:
            numframes = list(map(int, params[4:]))
        else:
            if "^" in roi:
                numframes = list(map(int, params[4:])) + [total]
            elif "!" in roi:
                numframes = [1] + list(map(int, params[4:]))
            else:
                numframes = [1, total]

        for i in range(numframes[0] - 1, numframes[1]):
            mask_im = masks[i]
            draw = ImageDraw.Draw(mask_im)
            if "ellipse" in roi:
                draw.ellipse((x1, y1, x2, y2), fill=255)
            else:
                draw.rectangle((x1, y1, x2, y2), fill=255)

    vid = cv2.VideoCapture(vidpath)
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                             vid.get(cv2.CAP_PROP_FPS), frame.shape[:2])
    color = Color(color)
    color = (color.get_blue() * 255, color.get_green() * 255, color.get_red()*255)
    bg = Image.new('RGB', frame.shape[:2], tuple(map(int, color)))
    for mask in masks:
        _, frame = vid.read(cv2.IMREAD_UNCHANGED)
        if mask == Image.new("L", frame.shape[:2], 0):
            masked = frame
        else:
            masked = bg.copy()
            masked.paste(Image.fromarray(frame), (0, 0), mask)
            masked = np.array(masked)
        writer.write(masked)
    vid.release()

    return 1

if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-v', '--vid', type=str, help='path to video')

    parser.add_argument('-c', '--color', type=str, default="black",
                        help='color written as a word like pink, aqua, etc.')
    parser.add_argument('-r', '--r', type=str, default=[], action='append',
                        help='LEFT,TOP,WIDTH,HEIGHT[;SHAPE=rect][^FRAME_START=1][!FRAME_FINISH=LAST_FRAME]]')
    parser.add_argument('-f', '--filename', type=str, help='name for a processed video')
    opt = parser.parse_args()#'-v E:\\100testimages.mp4 -c aqua -r 700,800,300,200;ellipse!20 -r 500,400,320,240^18!28 -f new.mp4'.split())

    roi_processing(*vars(opt).values())
