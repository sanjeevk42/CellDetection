from shapely.geometry import Polygon

import cv2
import numpy as np


class Rectangle:
    def __init__(self, bounds):
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.polygon = Polygon(self.points())

    def points(self):
        return [(self.minx, self.miny), (self.maxx, self.miny), (self.maxx, self.maxy), (self.minx, self.maxy)]

    def intersection(self, rectangle):
        polygon2 = Polygon(rectangle.points());
        return self.polygon.intersection(polygon2).area.real

    def union(self, rectangle):
        polygon2 = Polygon(rectangle.points());
        return self.polygon.union(polygon2).area.real

    def area(self):
        return self.polygon.area


def intersection(bounds1, bounds2):
    r1 = Rectangle(bounds1)
    r2 = Rectangle(bounds2)
    return r1.intersection(r2)


def union(bounds1, bounds2):
    r1 = Rectangle(bounds1)
    r2 = Rectangle(bounds2)
    return r1.union(r2)


def inter_over_union(bounds1, bounds2):
    r1 = Rectangle(bounds1)
    r2 = Rectangle(bounds2)
    return r1.intersection(r2) / r1.union(r2)


def get_annotated_img(img, annotations, bbox_size):
    annotated_img = np.array(img)
    for ann in annotations:
        x, y, s = ann
        cv2.rectangle(annotated_img, (x - bbox_size[0], y - bbox_size[1]), (x + bbox_size[0], y + bbox_size[1]),
                      (255, 0, 0), 2)
    return annotated_img


def inside_bound(bound, x, y):
    minx, miny, maxx, maxy = bound
    return minx <= x <= maxx and miny <= y <= maxy


def area(bound):
    r = Rectangle(bound)
    return r.area()


def feature_to_annotations(input_img, label_map, bbox_map):
    '''
    Converts grid based predictions to image co-ordinates.
    '''
    image_shape = input_img.shape
    label_shape = label_map.shape
    grid_size = np.array([image_shape[0] / label_shape[0], image_shape[1] / label_shape[1]])
    annotations = []
    for i in range(label_map.shape[0]):
        for j in range(label_map.shape[1]):
            grid_center = grid_size * (i, j) + grid_size / 2
            if label_map[i][j] > 0:
                x, y, s = grid_center[0], grid_center[1], label_map[i][j]
                bbox_loc = bbox_map[i][j]
                bbox_loc_abs = map(int, tuple(bbox_loc + np.concatenate([grid_center, grid_center])))
                annotations.append((x, y) + tuple(bbox_loc_abs))
    return annotations


def draw_prediction(input_img, labels, bboxes):
    '''
    Draw predicted bounding boxes on the image.
    '''
    ann_img = np.array(input_img)
    no_of_predictions = len(labels)
    for i in range(no_of_predictions):
        minx, miny, maxx, maxy, confidence = map(int, bboxes[i])
        if labels[i] == 1:
            cv2.rectangle(ann_img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        else:
            cv2.rectangle(ann_img, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
    return ann_img

def grid_patch_to_annoations(input_img, label_map, bbox_map):
    img_shape = input_img.shape
