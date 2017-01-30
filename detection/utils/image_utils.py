from shapely.geometry import Polygon

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters


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


def get_annotated_img(img, annotations, bbox_size, color=(255, 0, 0)):
    annotated_img = np.array(img)
    for ann in annotations:
        x, y = map(int, ann[0:2])
        cv2.rectangle(annotated_img, (x - bbox_size[0], y - bbox_size[1]), (x + bbox_size[0], y + bbox_size[1]),
                      color, 2)
    return annotated_img


def get_bbox(annotation, bbox):
    x, y, s = annotation
    return x - bbox[0], y - bbox[1], x + bbox[0], y + bbox[1]


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
            if label_map[i][j] > 0 and area():
                x, y, s = grid_center[0], grid_center[1], label_map[i][j]
                bbox_loc = bbox_map[i][j]
                bbox_loc_abs = map(int, tuple(bbox_loc + np.concatenate([grid_center, grid_center])))
                if area(bbox_loc_abs) > 100:
                    annotations.append((x, y) + tuple(bbox_loc_abs))
    return annotations


def draw_prediction(input_img, labels, bboxes):
    '''
    Draw predicted bounding boxes on the image.
    '''
    ann_img = np.array(input_img)
    no_of_predictions = len(labels)
    img_shape = ann_img.shape
    img_shape = img_shape[:2]
    for i in range(no_of_predictions):
        bbox = bboxes[i][:4] * (img_shape + img_shape)
        minx, miny, maxx, maxy = map(int, bbox)
        confidence = bboxes[i][4]
        if confidence > .5 and area(bbox) > 100:
            cv2.rectangle(ann_img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        else:
            pass
            # cv2.rectangle(ann_img, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
    return ann_img


def grid_patch_to_annoations(input_img, label_map, bbox_map):
    img_shape = input_img.shape


def gaussian_kernel(filter_size, sigma, mean):
    kx = cv2.getGaussianKernel(filter_size[0], sigma)
    ky = cv2.getGaussianKernel(filter_size[1], sigma)
    k = kx * np.transpose(ky)
    k *= (mean / np.max(k))
    return k


def local_maxima(data, neighborhood_size, threshold):
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
    yx = np.array(xy)
    yx[:, 0] = xy[:, 1]
    yx[:, 1] = xy[:, 0]
    return yx


def draw(img, points):
    width = 3
    ann_img = np.array(img)
    for point in points:
        ann_img[int(point[0]) - width:int(point[0]) + width, int(point[1]) - width:int(point[1]) + width] = [0, 255, 0]
    return ann_img


def filename_to_id(filename):
    fname = filename.split('.')[0]
    idx = None
    for x in fname.split('_'):
        try:
            idx = int(x)
        except ValueError:
            continue
    return idx


if __name__ == '__main__':
    print filename_to_id('cam0_0089.jpg')
    print filename_to_id('00220_bw.png')
    image = np.array(cv2.imread('/data/lrz/hm-cell-tracking/annotations/in/cam0_0314.jpg'), dtype=np.float64)
    print local_maxima(image[:, :, 0], 20, 0.5)
    # k = gaussian_kernel((41, 41), 10, 1)
    # plt.imshow(k, interpolation='nearest')
    # plt.show()
    # print(np.max(k))
