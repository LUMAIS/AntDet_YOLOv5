#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
from json import load

feature2num = {}


def shorten_file(jsFile: str) -> list:
    """
    Gets only relevant information from annotation file

    Args:
        jsFile (list): json file which contains annotations
    Returns:
        list of {<feature_id>: [<bbox>, <color>, <keyframe>]} for jsfile
    """
    annotList = []
    Annotation = namedtuple('Annotation', 'bbox color keyframe')

    for frame in jsFile:
        frameNum = frame['frameNumber'] - 1
        annotList.append(dict())
        for obj in frame['objects']:
            feature_id = obj['featureId']
            annotList[frameNum][feature_id] = Annotation(obj['bbox'], obj['color'], obj['keyframe'])
    return annotList


class Ant:
    def __init__(self, body, head, num=1):
        self.body = body
        self.head = head
        self.appears = num

    def print(self):
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


def valid(jsfile):
    global feature2num
    annot = shorten_file(jsfile)
    anthill = AntList()
    notsure = AntList()
    feature2num = {}
    heads = set()
    bodies = set()
    for frame in jsfile:
        body_num = 1
        head_num = 1
        for object in frame['objects']:
            if object['value'] == 'ant':
                bodies.add(object['featureId'])
                feature2num[object['featureId']] = body_num
                body_num +=1
            elif object['value'] == 'ant-head':
                heads.add(object['featureId'])
                feature2num[object['featureId']] = head_num
                head_num += 1
    for frame in annot:
        for bodyId in bodies:
            flag = True  # if True append ant to main list
            ant = None
            try:
                bLeft, bRight = frame[bodyId].bbox['left'], frame[bodyId].bbox['left'] + frame[bodyId].bbox['width']
                bTop, bBottom = frame[bodyId].bbox['top'], frame[bodyId].bbox['top'] + frame[bodyId].bbox['height']
                for headId in heads:
                    hCenter_x = frame[headId].bbox['left'] + frame[headId].bbox['width'] / 2
                    hCenter_y = frame[headId].bbox['top'] + frame[headId].bbox['height'] / 2
                    if bLeft < hCenter_x < bRight and bTop < hCenter_y < bBottom:
                        # if ant with this body on current frame exists turn flag False
                        flag *= 1 if ant is None else 0
                        ant = Ant(bodyId, headId)

                        # if head is already used but by a different ant turn flag to False
                        flag *= 0 if anthill.ishead(ant.head) and anthill.index(ant) == -1 else 1
                        if not flag:
                            notsure.append(ant)
                if flag and ant:
                    anthill.append(ant)
            except Exception as e:
                pass #print(e)
    # choosing best candidates from notsure list
    for i in range(len(notsure)):
        best = i
        ant1 = notsure[best]
        for j, ant2 in enumerate(notsure):
            if ant1.body == ant2.body and ant2.appears >= ant1.appears and not anthill.isbody(
                    ant2.body) and not anthill.ishead(ant2.head):
                best = j
                ant1 = ant2
        if not anthill.isbody(ant1.body) and not anthill.ishead(ant1.head):
            anthill.append(notsure[best])
    for ant in anthill:
        bodyId, headId = ant.body, ant.head
        for frame in jsfile:
            for obj in frame['objects']:
                if obj['featureId'] == bodyId:
                    bLeft, bRight = obj['bbox']['left'], obj['bbox']['left'] + obj['bbox']['width']
                    bTop, bBottom = obj['bbox']['top'], obj['bbox']['top'] + obj['bbox']['height']
                    break
            for obj in frame['objects']:
                if obj['featureId'] == headId:
                    hLeft, hRight = obj['bbox']['left'], obj['bbox']['left'] + obj['bbox']['width']
                    hTop, hBottom = obj['bbox']['top'], obj['bbox']['top'] + obj['bbox']['height']
                    if hLeft < bLeft or hTop < bTop or hRight > bRight or hBottom > bBottom:
                        # print('Annotations for body={}, head={} where changed on frame {}'.format(feature2num[bodyId],
                        #                                                                           feature2num[headId],
                        #                                                                           frame['frameNumber']))
                        obj['bbox']['left'] = max(hLeft, bLeft)
                        obj['bbox']['width'] = min(hRight, bRight) - hLeft
                        obj['bbox']['top'] = min(hTop, bTop)
                        obj['bbox']['height'] = max(hBottom, bBottom) - hTop
                        break


    return anthill


if __name__ == '__main__':
    al = AntList([Ant(1, 1), Ant(2, 1), Ant(3, 3)])
    al.append(Ant(2, 1))

    with open('E:\\work\\original_3-38_3-52.json', 'r') as file:
        jsfile = load(file)

    file = jsfile[:175]
    valid(file).print()
