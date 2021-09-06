import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"ant": 0,
                            "ant-head": 1,
                            "trophallaxis-ant": 2,
                            "larva": 3,
                            "trophallaxis-larva": 4,
                            "food-noise": 5,
                            "pupa": 6,
                            "barcode": 7} #,"uncategorized": 8}
# we delete uncategorized from possible classes to simplify the process of omitting them


# Convert the frame dict to the required yolo format and write it to disk
def convert_to_yolo(jsfile, img_size, filename, outdir):
    """
    jsfile: list from loaded json file
    img_size: tuple = (width, height) of the image/frame
    filename: future name of each txt file will take it as a beginning
    outdir: output directory for saving txt files
    """
    # go to an output directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, outdir))

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
                    image_w, image_h = img_size
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
        print("\n".join(print_buffer), file=open('{}_{}.txt'.format(filename, framenum), "w"))

#counts number of modified objects on frames, which were listed in the keyframes
def count_objects(jsfile, keyframes, obj_cost):
    """
    jsfile: list from loaded json file
    keyframes: string <n1>-<n2>,<n3>-<n4>,<n5>...
    obj_cost: cost of 1 annotation
    :return: number of modified objects
    """

    #return class if it was deleted to convert json to YOLO
    if not class_name_to_id_mapping.get("uncategorized"):
        class_name_to_id_mapping["uncategorized"] = 8
    cls_count = {key: 0 for key in class_name_to_id_mapping}
    # pattern = r'\d+\-\d+,?|\d+'

    #find all the intervals or just separated frame numbers
    framelst = keyframes.split(',')
    for i, el in enumerate(framelst):
        framelst[i] = el.split('-')
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
    print("Total: {} \nCost: ${}".format(sum, sum*obj_cost))
    return sum

if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-json-path', '--filepath', nargs='+',
                        help='Path for json files')
    # parser.add_argument('-vid', '--vid-path', default=None,
    #                     help='Path for the video')
    parser.add_argument('-pname', '--picname', nargs='+',
                        help='Name for outcoming label files')

    #create group with mutually exclusive elements: framesize and keyframe-obj
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--frame-size', default=None, type=str, nargs='+',
                       help='The size format is WxH, for example: 800x600')
    parser.add_argument('-o', '--outp-dir', type=str,
                        default=os.path.join(os.getcwd(), 'labels'),
                        help='Output directory for the label files')

    parser.add_argument('--object-cost', help=SUPPRESS, default=0.03, type=float)
    group.add_argument('-k', '--keyframed-objects', default=None, type=str,
                       help='String, that specify range of frames to count modified objects')

    args = parser.parse_args()

    for i, filepath in enumerate(args.filepath):
        with open(filepath) as jsonFile:
            annotations = json.load(jsonFile)

        if args.keyframed_objects:
            count_objects(annotations, args.keyframed_objects, args.object_cost)
        else:
            fm_size = tuple(map(lambda y: int(y), args.frame_size[i].split('x')))
            filename = os.path.split(filepath)[1][:-5]
            convert_to_yolo(annotations, fm_size, args.picname[i], args.outp_dir)
