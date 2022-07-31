#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: converters between YOLO, Labelbox import and Labelbox export data formats.

:Author: (c) Valentyna Pryhodiuk <vpryhodiuk@lumais.com>
:Date: 2022-07-06
"""
import json
import os
import uuid
import cv2

from collections import defaultdict

schema_lookup = {0: 'ckty9dfw44f8h0y9w0cnje3yr', 1: 'ckty9dfw44f8j0y9w9jgo7zx4',
                 2: 'ckty9dfw54f8l0y9wb6ig7vu4', 3: 'ckty9dfw54f8n0y9wcrb65ies',
                 4: 'ckty9dfw54f8p0y9w4qxygs9m', 5: 'ckty9dfw64f8r0y9wcu08h7ca',
                 6: 'ckty9dfw64f8t0y9wb3lv6svx', 7: 'ckty9dfw64f8v0y9wfnwl2fht', 8: 'ckty9dfw64f8x0y9w4kske8vf'}

class2id = {"ant": 0,
            "ant-head": 1,
            "trophallaxis-ant": 2,
            "larva": 3,
            "trophallaxis-larva": 4,
            "food-noise": 5,
            "pupa": 6,
            "barcode": 7,
            "uncategorized": 8}


# ------------------------ YOLO -> Labelbox (new type) ------------------------

def convert_yn(yolopath: str, imgpath: str, datarow_id: str, startframe: int = 1, lastframe: int = None) -> list:
    """
    Creates annotations using txt file with model predictionsOF THE FIRST FRAME and map it
     to all frames according to Labelbox Import style from startframe to lastframe.

    Args:
        yolopath (str): path to the txt annotation file of the YOLO format
        imgpath (str): path to the annotated picture
        datarow_id (str): id of the data_row to add this annotation to
        startframe (int): the number of the frame, which was annotated in a current file
        lastframe (int): the number of the last frame, which was annotated in a current file

    Returns:
        json representation of a bounding box
    """
    annotations = []
    img = cv2.imread(imgpath)
    height, width = img.shape[:2]
    with open(yolopath, 'r') as f:
        for line in f:
            cls, center_x, center_y, w, h = line.split()  # tuple(map(lambda x: int(x), obj.split()))
            center_x, w = float(center_x) * width, float(w) * width
            center_y, h = float(center_y) * height, float(h) * height
            # cls, center_x, center_y, w, h = tuple(map(lambda x: float(x), line.split()))
            part = {"uuid": str(uuid.uuid4()),
                    "schemaId": schema_lookup[int(cls)],
                    "dataRow": {
                        "id": datarow_id
                    },
                    "segments": [
                        {
                            "keyframes": [
                                {
                                    "frame": startframe,
                                    "bbox": {
                                        "top": center_y - h / 2,
                                        "left": center_x - w / 2,
                                        "height": h,
                                        "width": w
                                    }
                                }]
                        }]
                    }
            if lastframe:
                part["segments"][0]['keyframes'] += [{
                    "frame": lastframe,
                    "bbox": {
                        "top": center_y - h / 2,
                        "left": center_x - w / 2,
                        "height": h,
                        "width": w
                    }
                }]
            annotations.append(part)

    return annotations


# ------------------------ YOLO -> Labelbox (old type) ------------------------

# NOT the case when we have featureIds
def convert_yo(dirpath: str, imgpath: str) -> list:
    """
    Creates annotations using YOLO created annotations to convert them to the old Labelbox format.
    (to use orbAnalysis app to track id's)

    Args:
        dirpath (str): path to a directory with txt annotation files (YOLO format)
        imgpath (str): path to the annotated picture (to take dimensions)

    Returns:
        json representation of a whole video
    """
    classes = {val: key for key, val in class2id.items()}
    cur = os.getcwd()
    os.chdir(dirpath)
    filenames = next(os.walk(dirpath), (None, None, []))[2]
    annotations = []
    img = cv2.imread(imgpath)
    height, width = img.shape[:2]
    filenames.sort()
    print(filenames)

    for file in filenames:
        with open(file, 'r') as f:
            framenum = int(file.rstrip('.txt').split('_')[-1])
            part = {"frameNumber": framenum, "objects": []}
            for line in f:
                cls, center_x, center_y, w, h = line.split()  # tuple(map(lambda x: int(x), obj.split()))
                center_x, w = float(center_x) * width, float(w) * width
                center_y, h = float(center_y) * height, float(h) * height
                # cls, center_x, center_y, w, h = tuple(map(lambda x: float(x), line.split()))
                part["objects"].append({"schemaId": schema_lookup[int(cls)],
                                        "title": classes[int(cls)],
                                        "value": classes[int(cls)],
                                        "bbox": {
                                            "top": center_y - h / 2,
                                            "left": center_x - w / 2,
                                            "height": h,
                                            "width": w
                                        },
                                        "classifications": []
                                        })
            annotations.append(part)

    return annotations


# ------------------------ Labelbox (new type) -> Labelbox (old type) ------------------------

def convert_no(filepath: str) -> list:
    """
    Creates annotations using YOLO created annotations to convert them to the old Labelbox format.

    Args:
        filepath (str): path to a file (Labelbox new)

    Returns:
        json representation of a whole video
    """
    res = defaultdict(list)  # {"frameNumber": i, "objects": []} for i in range(1, )]
    schema2cls = {val: key for key, val in class2id.items()}  # schema_lookup.items()}
    schema2cls = {schema_lookup[key]: val for key, val in schema2cls.copy().items()}

    with open(filepath, 'r') as file:
        annotations = json.load(file)

    def scale(bbox1, bbox2, l1, l2):
        bbox = dict()
        for k in bbox2:
            bbox[k] = (float(bbox1[k]) * l1 + float(bbox2[k]) * l2) / (l1 + l2)
        return bbox

    for obj in annotations:
        featureId = str(uuid.uuid4())
        for an in obj["segments"]:
            keyframe1 = an['keyframes'][0]
            keyframe2 = keyframe1 if len(an['keyframes']) == 1 else an['keyframes'][1]
            fstart = keyframe1['frame']
            fend = fstart if len(an['keyframes']) == 1 else keyframe2['frame']
            for i in range(fstart, fend + 1):
                res[i] += [
                    {
                        'featureId': featureId,
                        'bbox': scale(keyframe1['bbox'], keyframe2['bbox'], fend - i + 1, i - fstart),
                        'schemaId': obj['schemaId'],
                        'title': schema2cls[obj['schemaId']],
                        'value': schema2cls[obj['schemaId']]
                    }
                ]

    result = [{"frameNumber": k, "objects": v} for k, v in dict(res).items()]

    return result
