from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import cv2
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

col = {"0": (90, 237, 253),
        "1": (249, 64, 187),
        "2": (255, 0, 0),
        "3": (52, 134, 31),
        "4": (51, 0, 102),
        "5": (204, 0, 204),
        "6": (153, 153, 0),
        "7": (0, 0, 153)} #,"uncategorized": 8}

def visualize_bbox(image: np.ndarray, line: str, sframe: tuple, col: tuple) -> np.ndarray:
    """
    Draws a bounding box on an image

    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
    Returns:
        image with a bounding box drawn on it.
    """
    # start = (int(tool["bbox"]["left"]), int(tool["bbox"]["top"]))
    # end = (int(tool["bbox"]["left"] + tool["bbox"]["width"]),
    #        int(tool["bbox"]["top"] + tool["bbox"]["height"]))
    cl, center_x, center_y, width, height = line.split()
    # image_w = width
    # image_h = height
    top = (float(center_x) - float(width)/2)*sframe[0]
    left = (float(center_y) - float(height)/2)*sframe[1]
    down = (float(center_x) + float(width)/2)*sframe[0]
    right = (float(center_y) + float(height)/2)*sframe[1]
    start = (int(top), int(left))
    end = (int(down), int(right))
    return cv2.rectangle(image, start, end, col, 1)

def surf_files(txtdir, viddir, sframe):
    os.chdir(viddir)
    mapping = {"EuresysCapturing_IR_100_2021-08-24_17": [],  # "100testimages_s25p_f10.mp4",
               "Maxvision_Camera_Video_Sample_#6-galal": []}  # "6.mp4"}
    f = []
    for (dirpath, dirnames, filenames) in os.walk(txtdir):
        f.extend(filenames)
        break

    for labelfile in f:
        name = os.path.split(labelfile)[1][:-4]
        num = int(name.split('_')[-1])
        name = '_'.join(name.split('_')[:-1])
        mapping[name].append(num)
        mapping[name].sort()

    os.chdir(txtdir)
    for name, framenum in mapping.items():
        for i in range(1, max(framenum)+1):
            vid = cv2.VideoCapture(name)
            success, image = vid.read()
            image = image[:, :, ::-1]
            if i in framenum:
                with open("{}_{}.txt".format(name, i)) as file:
                    for line in file.readlines():
                        image = visualize_bbox(image.astype(np.uint8), line, sframe, col=col[line[0]])
                plt.figure(1)
                plt.imshow(image)
                plt.title('frameNumber ' + str(i))
                plt.pause(10)
                plt.clf()
            if success:
                clear_output(wait=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-txt-path', '--txtdir', nargs='+',
                        help='Path for txt files')
    parser.add_argument('-vid', '--viddir', nargs='+',
                        help='Path for the video')

    #create group with mutually exclusive elements: framesize and keyframe-obj
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--frame-size', default=None, type=str,
                       help='The size format is WxH, for example: 800x600')
    # parser.add_argument('-o', '--outp-dir', type = str,
    #                     default=os.path.join(os.getcwd(), 'labels'),
    #                     help='Output directory for the label files')

    args = parser.parse_args()
    surf_files(args.txtdir, args.viddir, args.frame_size)




