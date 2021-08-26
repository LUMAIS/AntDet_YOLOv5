import os
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"ant": 0,
                            "ant-head": 1,
                            "trophallaxis-ant": 2,
                            "larva": 3,
                            "trophallaxis-larva": 4,
                            "food noise": 5,
                            "pupa": 6,
                            "barcode": 7} #, "uncategorized": 8}
# we delete uncategorized from possible classes to simplify the process of omitting them


# Convert the frame dict to the required yolo format and write it to disk
def convert_to_yolo(jsfile, img_size, outdir):
    """
    jsfile: json file
    img_size: tuple = (height, width) of the image/frame
    outdir: output directory for saving txt files
    """
    # go to an output directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, outdir))

    for frame in jsfile:
        print_buffer = []

        # For each object on a frame
        for obj in frame['objects']:
            try:
                class_id = class_name_to_id_mapping[obj["title"]]
                b = obj['bbox']
                #be sure, that no low-confidence objects will be present in our training set
                if not obj['classifications'] or obj['classifications'][0]['answers'][0]['value'] != 'low-confidence':
                    # Transform the bbox coordinates as per the format required by YOLO v5
                    b_center_x = b["left"] + b["width"]/ 2
                    b_center_y = b["top"] + b["height"]/ 2
                    b_width = b["width"]
                    b_height = b["height"]

                    # Normalise the coordinates by the dimensions of the image
                    image_h, image_w = img_size
                    b_center_x /= image_w
                    b_center_y /= image_h
                    b_width /= image_w
                    b_height /= image_h

                    # Write the bbox details to the file
                    print_buffer.append(
                        "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
            except KeyError: #made to ommit uncategorized and wrong class names
                pass
        # print("Invalid Class or uncategorized")

        framenum = str(frame["frameNumber"])
        # Save the annotation to disk
        print("\n".join(print_buffer), file=open(framenum + ".txt", "w"))

if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-json-path', '--filepath', default="EuresysCapturing_IR_100_2021-08-24_17.json",
                        help='Path for json file')
    # parser.add_argument('-vid', '--vid-path', default=None,
    #                     help='Path for the video')
    parser.add_argument('-o', '--outp-dir', default='labels',
                        help='Output directory for the label files')
    args = parser.parse_args()
    with open(args.filepath) as jsonFile:
        annotations = json.load(jsonFile)
        jsonFile.close()
    img_size = (1000, 1000)
    convert_to_yolo(annotations, img_size, args.outp_dir)
