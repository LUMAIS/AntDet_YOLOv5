#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
from json import load, dump
from typing import Dict

feature2num = {}


def shorten_file(jsFile: str) -> Dict[str, list]:
    """
    Gets only relevant information from annotation file

    Args:
        jsFile (list): json file which contains annotations
    Returns:
        {framenum: {<feature_id>: [<bbox>, <color>, <keyframe>]}} for jsfile
    """
    annotDict = dict()
    Annotation = namedtuple('Annotation', 'bbox color keyframe')

    for frame in jsFile:
        frameNum = frame['frameNumber']
        annotDict[frameNum] = dict()
        for obj in frame['objects']:
            feature_id = obj['featureId']
            annotDict[frameNum][feature_id] = Annotation(obj['bbox'], obj['color'], obj['keyframe'])
    return annotDict


class Ant:
    def __init__(self, body, head, num=1):
        self.body = body
        self.head = head
        self.appears = num

    def print(self):
        # print("Ant(body = {}, head = {}, appears = {})".format(self.body, self.head, self.appears))
        print("Ant(body = {}, head = {}, appears = {})".format(feature2num[self.body],
                                                               feature2num[self.head],
                                                               self.appears))


class AntList(list):
    def ishead(self, head):
        for ant in self:
            if ant.head == head:
                return True
        return False

    def isbody(self, body):
        for ant in self:
            if ant.body == body:
                return True
        return False

    def index(self, ant):
        for ind, el in enumerate(self):
            if ant.body == el.body and ant.head == el.head:
                return ind
        return -1

    def append(self, *args):
        for el in args:
            ind = self.index(el)
            if ind != -1:
                self[ind].appears += el.appears
            else:
                super().append(el)

    def print(self):
        for ant in self:
            ant.print()


def valid(jsfile, roifile=[]):
    global feature2num
    annot = shorten_file(jsfile)
    anthill = AntList()
    notsure = AntList()
    heads = set()
    bodies = set()
    for frame in jsfile:
        body_num = 1
        head_num = 1
        for object in frame['objects']:
            if object['value'] == 'ant':
                bodies.add(object['featureId'])
                feature2num[object['featureId']] = body_num if not object['featureId'] in feature2num else max(body_num,
                                                                                                               feature2num[
                                                                                                                   object[
                                                                                                                       'featureId']])
                body_num += 1
            elif object['value'] == 'ant-head':
                heads.add(object['featureId'])
                feature2num[object['featureId']] = head_num if not object['featureId'] in feature2num else max(head_num,
                                                                                                               feature2num[
                                                                                                                   object[
                                                                                                                       'featureId']])
                head_num += 1

    for frame in annot.values():
        for bodyId in bodies:
            if bodyId in frame:
                bLeft, bRight = frame[bodyId].bbox['left'], frame[bodyId].bbox['left'] + frame[bodyId].bbox['width']
                bTop, bBottom = frame[bodyId].bbox['top'], frame[bodyId].bbox['top'] + frame[bodyId].bbox['height']
                for headId in heads:
                    if headId in frame:
                        hCenter_x = frame[headId].bbox['left'] + frame[headId].bbox['width'] / 2
                        hCenter_y = frame[headId].bbox['top'] + frame[headId].bbox['height'] / 2
                        if bLeft < hCenter_x < bRight and bTop < hCenter_y < bBottom:
                            ant = Ant(bodyId, headId)
                            notsure.append(ant)
    for _ in range(3):
        for bodyId in bodies:
            best = AntList([Ant(bodyId, '-0', 0)])
            for ant in notsure.copy():
                if ant.body == best[0].body and ant.appears > best[0].appears:
                    try:
                        for j in best:
                            notsure.remove(j)
                    except ValueError as e:
                        pass
                    best = AntList([ant])
                elif ant.body == best[0].body and ant.appears == best[0].appears:
                    best.append(ant)
                elif ant.body == best[0].body:
                    notsure.remove(ant)

            if len(best) > 1:
                for j in best.copy():
                    if anthill.ishead(j.head):
                        best.remove(j)
            if len(best) == 1 and (not anthill.ishead(best[0].head)) and best != AntList([Ant(bodyId, '0', 0)]):
                try:
                    for j in best:
                        notsure.remove(j)
                    anthill += best
                except ValueError as e:
                    pass

    print('The number of ants detected by script', len(anthill))

    for ant in anthill:
        bodyId, headId = ant.body, ant.head
        for frame in jsfile:
            for obj in frame['objects']:
                if obj['featureId'] == bodyId:
                    bLeft, bRight = obj['bbox']['left'], obj['bbox']['left'] + obj['bbox']['width']
                    bTop, bBottom = obj['bbox']['top'], obj['bbox']['top'] + obj['bbox']['height']
                    for roi in roifile:
                        if roi.interval[0] <= frame['frameNumber'] <= roi.interval[1]:
                            x1, y1, w, h = roi.ROI[0], roi.ROI[1]
                            bLeft, bRight = max(bLeft, x1), min(bRight, x1 + w)
                            bTop, bBottom = max(bTop, y1), min(bBottom, y1 + h)

            for obj in frame['objects']:
                if obj['featureId'] == headId:
                    hLeft, hRight = obj['bbox']['left'], obj['bbox']['left'] + obj['bbox']['width']
                    hTop, hBottom = obj['bbox']['top'], obj['bbox']['top'] + obj['bbox']['height']

                    if not bTop < (hBottom + hTop) / 2 < bBottom or not bLeft < (hLeft + hRight) / 2 < bRight:
                        print(
                            'WARNING: flying head featureId {}, body featureId {} probably the number {}, frame {}'.format(
                                headId,
                                bodyId,
                                feature2num[headId],
                                frame['frameNumber']))
                    elif hLeft < bLeft or hTop < bTop or hRight > bRight or hBottom > bBottom:
                        # print('Annotations for body={}, head={} where changed on frame {}'.format(feature2num[bodyId],
                        #                                                                           feature2num[headId],
                        #                                                                           frame['frameNumber']))
                        obj['bbox']['left'] = max(hLeft, bLeft)
                        obj['bbox']['width'] = min(hRight, bRight) - hLeft
                        obj['bbox']['top'] = max(hTop, bTop)
                        obj['bbox']['height'] = min(hBottom, bBottom) - hTop
                    for roi in roifile:
                        if roi.interval[0] <= frame['frameNumber'] <= roi.interval[1]:
                            x1, y1, w, h = roi.ROI[0], roi.ROI[1]
                            hLeft, hRight = max(hLeft, x1), min(hRight, x1 + w)
                            hTop, hBottom = max(hTop, y1), min(hBottom, y1 + h)
    with open("new.json", 'w') as file:
        dump(jsfile, file)
    return anthill


if __name__ == '__main__':
    parser = ArgumentParser(description='Document Taxonomy Builder.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-a', '--annotations', type=str, help='Path to an annotation file')
    parser.add_argument('-r', '--roi', type=str, help='Path to the ROI file')
    opt = parser.parse_args() #"-a E:\\work\\EuresysCapturing_IR_100_2021-08-24_17.json".split()
    if opt.annotations:
        with open(opt.annotations, 'r') as file:
            jsfile = load(file)
    if opt.roi:
        with open(opt.roi, 'r') as file:
            roifile = load(file)
    else:
        roifile = []
    valid(jsfile, roifile).print()
    # print(feature2num)
