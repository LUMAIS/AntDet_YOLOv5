import os
import numpy as np
from re import findall
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"ant": 0,
                            "ant-head": 1,
                            "trophallaxis-ant": 2,
                            "larva": 3,
                            "trophallaxis-larva": 4,
                            "food-noise": 5,
                            "pupa": 6,
                            "barcode": 7,
                            "uncategorized": 8}


# Convert the frame dict to the required yolo format and write it to disk
def convert_to_yolo(jsfile, img_size, outdir):
    """
    jsfile: list from loaded json file
    img_size: tuple = (height, width) of the image/frame
    outdir: output directory for saving txt files
    """
    # go to an output directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, outdir))

    # we delete uncategorized from possible classes to simplify the process of omitting them
    class_name_to_id_mapping.pop("uncategorized")

    for frame in jsfile:
        print_buffer = []

        # For each bounding box
        for obj in frame['objects']:
            try:
                class_id = class_name_to_id_mapping[obj["title"]]
                b = obj['bbox']
                flag = 1  #used to check if object has an attribute = low-confidence
                if obj['classifications']:
                    for cl in obj['classifications']:
                        for answer in cl['answers']:
                            flag *= 0 if answer['value'] == 'low-confidence' else 1
                if not obj['classifications'] or flag:
                    # Transform the bbox coordinates as per the format required by YOLO v5
                    b_center_x = b["left"] + b["width"] / 2
                    b_center_y = b["top"] + b["height"] / 2
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
                else:
                    # print(obj)
                    pass
            except KeyError:
                pass
        # print("Invalid Class or uncategorized")

        framenum = str(frame["frameNumber"])
        # Save the annotation to disk
        print("\n".join(print_buffer), file=open(framenum + ".txt", "w"))

#counts number of modified objects on frames, which were listed in the keyframes
def count_salary(jsfile, keyframes):
    """
    jsfile: list from loaded json file
    keyframes: string <n1>-<n2>,<n3>-<n4>,<n5>...
    :return: number of modified objects
    """

    #return class if it was deleted to convert json to YOLO
    if not class_name_to_id_mapping.get("uncategorized"):
        class_name_to_id_mapping["uncategorized"] = 8
    cls_count = {key: 0 for key in class_name_to_id_mapping}
    pattern = r'\d+\-\d+,?|\d+'
    digit = r'\d+'
    #find all the intervals or just separated frame numbers
    framelst = findall(pattern, keyframes) if keyframes else []
    for i, el in enumerate(framelst):
        framelst[i] = findall(digit, el)
    sum = 0
    print_buffer = []

    for interval in framelst:
        if len(interval) == 1:
            beginning, ending = interval[0], interval[0]
        else:
            beginning, ending = interval

        for i in range(int(beginning) - 1, int(ending)):
            frame = jsfile[i]
            print_buffer.append(frame["frameNumber"])

            # For each obj in frame
            for obj in frame['objects']:
                if obj['keyframe']: #if true, than changes where made
                    try:
                        cls_count[obj["title"]] += 1
                        sum += 1
                    except KeyError:
                        pass

    print_buffer.sort()
    print("Frames taken to account: ", print_buffer)
    for key, value in cls_count.items():
        print("{}: {}".format(key, value))
    print("Total: {} \nSalary: {}$".format(sum, sum*0.03))
    return sum

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
    parser.add_argument('-k', '--keyframed-objects', default=None, type=str,
                        help='String, that specify frames, where number of changed points should be counted')

    args = parser.parse_args()

    pattern = r'\d+\-\d+,?|\d+'
    if args.keyframed_objects and not findall(pattern, args.keyframed_objects):
        print("Something wrong with your frame list. Check if it looks like f1-f2,f3-f4,...,f(n)-f(n+1) or just separeted frames")
    else:
        with open(args.filepath) as jsonFile:
            annotations = json.load(jsonFile)
            jsonFile.close()
        img_size = (1000, 1000)
        if args.keyframed_objects:
            count_salary(annotations, args.keyframed_objects)
        else:
            convert_to_yolo(annotations, img_size, args.outp_dir)
