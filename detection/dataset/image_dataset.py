import os

import cv2
import numpy as np

from detection.utils import image_utils
from detection.utils.image_utils import get_annotated_img
from detection.utils.logger import logger

np.random.seed(43)


class ImageDataset(object):
    '''
    Maintains training dataset and implements generators which provide data while training.
    '''

    def __init__(self, base_dir, test_ratio=0.20, validation_ratio=0.10):
        self.base_dir = base_dir
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.annotation_file = os.path.join(base_dir, 'localizations.txt')
        self.all_frames = self.read_annotations()
        self.dataset_size = len(self.all_frames)
        self.split_dataset()

    class FrameInfo():
        '''
        Contains annotated image information such as image data, annotations etc.
        '''

        def __init__(self, base_dir, img_id, roi, annotations, bbox=(20, 20)):
            self.base_dir = base_dir
            self.img_id = img_id
            # add buffer to region of interest ...
            self.roi = roi[0] - bbox[0], roi[1] + bbox[0], roi[2] - bbox[1], roi[3] + bbox[1]
            self.bbox = bbox
            img_data = cv2.imread(os.path.join(self.base_dir, self.img_id))
            self.img_data = img_data[self.roi[2]: self.roi[3], self.roi[0]: self.roi[1]]
            # normalize annotations for cropped image...
            self.annotations = [(ann[0] - self.roi[0], ann[1] - self.roi[2], ann[2]) for ann in annotations]
            self.all_seq_patches = []

        def annotated_img(self):
            annotated_img = get_annotated_img(self.img_data, self.annotations, self.bbox)
            return annotated_img

        def get_random_patches(self, patch_size, no_of_patches):
            '''
            Randomly samples no_of_patches of patch_size from image.
            '''
            img_shape = self.frame_info.img_data.shape
            x = np.random.randint(0, img_shape[0] - patch_size[0], no_of_patches)
            y = np.random.randint(0, img_shape[1] - patch_size[1], no_of_patches)
            xy = zip(x, y)
            img_patches = []
            for i, j in xy:
                img_patch = ImageDataset.Patch(self, j, i, patch_size)
                img_patches.append(img_patch)
            return img_patches

        def sequential_patches(self, patch_size, step_size):
            '''
            Returns all sequential patches from image separated by step.
            '''
            if len(self.all_seq_patches) == 0:
                img_shape = self.img_data.shape
                x = range(0, img_shape[0] - patch_size[0], step_size[0])
                y = range(0, img_shape[1] - patch_size[1], step_size[1])
                xy = [(i, j) for i in x for j in y]
                img_patches = []
                for i, j in xy:
                    img_patch = ImageDataset.Patch(self, j, i, patch_size)
                    img_patches.append(img_patch)
                self.all_seq_patches = img_patches
            return self.all_seq_patches

    class Patch(object):
        '''
        Represents a patch inside an input image.
        '''

        def __init__(self, frame_info, startx, starty, patch_size):
            self.frame_info = frame_info
            self.startx = startx
            self.starty = starty
            self.patch_size = patch_size
            self.__find_annotations()

        def get_img(self):
            img_data = self.frame_info.img_data
            img_patch = img_data[self.starty:self.starty + self.patch_size[0],
                        self.startx:self.startx + self.patch_size[1]]
            return img_patch

        def __find_annotations(self):
            '''
            Finds annotations whose bounding box completely lie in the patch.
            '''
            annotations = []
            for ann in self.frame_info.annotations:
                x, y, s = ann
                bbox_size = self.frame_info.bbox
                # minx, miny, maxx, maxy = x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1]
                # if minx >= self.startx and maxx <= self.startx + self.patch_size[1] and miny >= \
                #         self.starty and maxy <= self.starty + self.patch_size[1]:
                if self.startx <= x <= self.startx + self.patch_size[1] and self.starty <= y <= self.starty + \
                        self.patch_size[1]:
                    annotations.append(ann)
            self.ann_relative = annotations
            self.annotations = [(ann[0] - self.startx, ann[1] - self.starty, ann[2]) for ann in annotations]

        def ann_patch(self):
            ann_patch = get_annotated_img(self.get_img(), self.annotations, self.frame_info.bbox)
            return ann_patch

    def split_dataset(self):
        '''
        Splits the dataset into training, testing and validation sets.
        '''
        idx = range(self.dataset_size)
        idx = np.random.permutation(idx)
        training_data_size = int(self.dataset_size * (1 - self.test_ratio))
        validation_size = int(training_data_size * self.validation_ratio)
        self.validation_idx = idx[:validation_size]
        self.training_idx = idx[validation_size:training_data_size]
        self.testing_idx = idx[training_data_size:]

        logger.info('Total dataset size:{}, training:{}, validation:{}, test:{}', self.dataset_size,
                    len(self.training_idx), len(self.validation_idx), len(self.testing_idx))

    def read_annotations(self):
        '''
        Reads the annotation file and create frame objects for all image frames.
        '''
        all_annotations = {}
        with open(self.annotation_file, 'r') as fr:
            for line in fr:
                frame, x, y, s, _ = list(map(int, line.split()))
                if frame not in all_annotations:
                    all_annotations[frame] = []

                all_annotations[frame].append((x, y, s))

        all_files = os.listdir(self.base_dir)
        all_files = [fn for fn in all_files if fn.endswith('_bw.png')]
        all_files.sort(key=lambda x: int(x[:5]))
        # since image files are listed sequentially in annotation file

        roi = self.get_global_bounds(all_annotations)

        frame_infos = []
        for i, fn in enumerate(all_files):
            annotations = all_annotations[i]
            frame_info = ImageDataset.FrameInfo(self.base_dir, fn, roi, annotations);
            frame_infos.append(frame_info)

        return frame_infos

    def get_global_bounds(self, all_annotations):
        '''
        Returns largest image region such that it covers complete annotated region in all input images.
        '''
        img_bounds = []
        for key, annotations in all_annotations.items():
            minx = min(annotations, key=lambda ann: ann[0])[0]
            maxx = max(annotations, key=lambda ann: ann[0])[0]
            miny = min(annotations, key=lambda ann: ann[1])[1]
            maxy = max(annotations, key=lambda ann: ann[1])[1]
            img_bounds.append((minx, maxx, miny, maxy))
        gminx = min(img_bounds, key=lambda x: x[0])[0]
        gmaxx = max(img_bounds, key=lambda x: x[1])[1]
        gminy = min(img_bounds, key=lambda x: x[2])[2]
        gmaxy = max(img_bounds, key=lambda x: x[3])[3]

        return gminx, gmaxx, gminy, gmaxy

    def grid_dataset_generator(self, batch_size, patch_size=(224, 224), grid_size=(28, 28)):
        '''
        Training data generator for grid based predictions.
        '''
        all_patches = []
        for idx in self.training_idx:
            all_patches.extend(self.all_frames[idx].sequential_patches(patch_size, (200, 200)))
        logger.info('Total patches:{}', len(all_patches))
        while True:
            # Randomly select patches from sequential patches ...
            patches_idx = np.random.randint(0, len(all_patches), batch_size)
            class_batch = []
            bbout_batch = []
            input_batch = []
            for idx in patches_idx:
                patch = all_patches[idx]
                label_map, bbox_map = self.grid_ground_truth(patch, grid_size)
                class_batch.append(label_map)
                bbout_batch.append(bbox_map)
                input_batch.append(patch.get_img())

            yield (
                {'input_1': np.array(input_batch)},
                {'class_out': np.array(class_batch), 'bb_out': np.array(bbout_batch)})

    def grid_ground_truth(self, patch, grid_size):
        '''
        Returns label and bounding box feature maps for a patch.
        '''
        patch_size = np.array(patch.patch_size)
        response_map_shape = list(patch_size / grid_size)
        bbox_map_shape = response_map_shape + [4]
        bbox_map = np.zeros(bbox_map_shape)
        label_map = np.zeros(response_map_shape + [1])

        bbox_size = patch.frame_info.bbox
        for i in range(int(bbox_map_shape[0])):
            for j in range(int(bbox_map_shape[1])):
                grid_index = np.array([i, j]) * grid_size
                grid_bound = tuple(grid_index) + tuple(grid_index + grid_size)
                best_annotation_idx = -1
                max_intersection = 0
                for idx, ann in enumerate(patch.annotations):
                    x, y, s = ann
                    ann_bound = (x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1])
                    intersection = image_utils.intersection(grid_bound, ann_bound)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        best_annotation_idx = idx
                if best_annotation_idx > 0:
                    grid_center = grid_index + np.array(grid_size) / 2
                    x, y, s = patch.annotations[best_annotation_idx]
                    ann_bound = np.array((x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1]))
                    ann_bound_norm = ann_bound - np.concatenate([grid_center, grid_center])
                    label_map[i][j] = 1  # currenly only binary segmentation is being done ...
                    bbox_map[i][j] = ann_bound_norm

        return label_map, bbox_map

    def grid_patch_dataset_generator(self, batch_size, patch_size=(224, 224), grid_size=(28, 28), nb_objects=5):
        '''
        Training data generator for grid patch based predictions where nb_objects labels and bounding boxes are
        predicted at each grid.
        '''
        all_patches = []
        for idx in self.training_idx:
            all_patches.extend(self.all_frames[idx].sequential_patches(patch_size, (200, 200)))
        logger.info('Total patches:{}', len(all_patches))
        while True:
            # Randomly select patches from sequential patches ...
            patches_idx = np.random.randint(0, len(all_patches), batch_size)
            class_batch = []
            bbout_batch = []
            input_batch = []
            for idx in patches_idx:
                patch = all_patches[idx]
                label_map, bbox_map = self.grid_patch_ground_truth(patch, grid_size, nb_objects)
                class_batch.append(label_map)
                bbout_batch.append(bbox_map)
                input_batch.append(patch.get_img())

            yield (
                {'input_1': np.array(input_batch)},
                {'class_out': np.array(class_batch), 'bb_out': np.array(bbout_batch)})

    def grid_patch_ground_truth(self, patch, grid_size, nb_objects):
        '''
        Returns ground truth labels and bounding box such that nb_objects are predicted at each grid.
        '''
        response_map_shape = list(np.array(patch.patch_size) / grid_size)
        label_map = np.zeros(response_map_shape + [nb_objects])
        bbox_map = np.zeros(response_map_shape + [nb_objects, 5])
        patch_annotations = patch.annotations
        bbox_size = patch.frame_info.bbox
        for i in range(response_map_shape[0]):
            for j in range(response_map_shape[1]):
                grid_index = np.array([i, j]) * grid_size
                grid_bounds = tuple(grid_index) + tuple(grid_index + grid_size)
                grid_annotations = [ann for ann in patch_annotations if
                                    grid_bounds[0] <= ann[0] <= grid_bounds[2] and grid_bounds[1] <= ann[1] <=
                                    grid_bounds[3]]

                grid_annotations.sort(
                    key=lambda x: image_utils.intersection(grid_bounds, self.get_bbox(x, bbox_size)),
                    reverse=True)

                grid_ann_length = len(grid_annotations)
                for ann_idx in range(min(grid_ann_length, nb_objects)):
                    x, y, s = grid_annotations[ann_idx]
                    confidence = image_utils.intersection(grid_bounds, self.get_bbox(ann, bbox_size))
                    obj_center = np.array([x, y]) - grid_index
                    bbox_map[i][j][ann_idx] = np.concatenate([obj_center, bbox_size, [confidence]])
                    label_map[i][j][ann_idx] = 1

            return label_map, bbox_map

    def patch_dataset_generator(self, batch_size, patch_size=(64, 64), no_of_objects=5, sampling_type='random'):
        '''
        Dataset generator for no_of_objects predictions in image patch..
        '''
        # Only random sampling of image patches is implemented
        all_patches = []
        for frame_info in self.all_frames:
            all_patches.extend(frame_info.sequential_patches(patch_size, (50, 50)))
        logger.info('Total patches :{}', len(all_patches))
        while True:
            patches_idx = np.random.randint(0, len(all_patches), batch_size)
            image_batch = []
            class_map = []
            bbox_map = []
            for patch_idx in patches_idx:
                patch = all_patches[patch_idx]
                image_batch.append(patch.get_img())
                label, bbox = self.patch_ground_truth(patch, no_of_objects)
                class_map.append(label)
                bbox_map.append(bbox)
            yield ({'input_1': np.array(image_batch)}, {'class_out': np.array(class_map), 'bb_out': np.array(bbox_map)})

    def patch_ground_truth(self, patch, no_of_objects):
        '''
        Returns objects which have .5 fraction of intersecting area in the patch image.
        '''
        class_labels = []
        bbox_size = patch.frame_info.bbox
        patch_bound = (0, 0) + patch.patch_size
        labels = np.zeros(no_of_objects)
        bboxes = np.zeros((no_of_objects, 5))
        patch_annotations = list(patch.annotations)
        patch_annotations.sort(key=lambda x: image_utils.intersection(patch_bound, self.get_bbox(x, bbox_size)),
                               reverse=True)
        no_of_annotations = len(patch_annotations)
        for i in range(min(no_of_objects, no_of_annotations)):
            ann_bound = self.get_bbox(patch_annotations[i], bbox_size)
            bboxes[i] = ann_bound + (image_utils.intersection(patch_bound, ann_bound),)
            labels[i] = 1
        return labels, bboxes

    def get_bbox(self, annotation, bbox):
        x, y, s = annotation
        return x - bbox[0], y - bbox[1], x + bbox[0], y + bbox[1]


if __name__ == '__main__':
    dataset_gen = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations')
    # for x in dataset_gen.patch_dataset_generator(1):
    #     print x
    #     break
    # for frame in dataset_gen.all_frames:
    #     if frame.img_id == '00402_bw.png':
    #         break
    # patches = frame.sequential_patches((64, 64), (60, 60))
    for data in dataset_gen.grid_patch_dataset_generator(200, patch_size=(224, 224), grid_size=(32, 32), nb_objects=5):
        img_data = data[0]['input_1']
        labels, bboxes = data[1]['class_out'], data[1]['bb_out']
        #print img_data.shape, labels.shape, bboxes.shape
        # for i in range(len(img_data)):
        #     img_ann = image_utils.draw_prediction(patch.get_img(), labels, bboxes)
        # cv2.imwrite('/data/patches/pred{}.png'.format(i), img_ann)
        # label_map, bbox_map = dataset_gen.grid_ground_truth(patches[0], (28, 28))
        # print label_map.shape, bbox_map.shape
        # annotations = image_utils.feature_to_annotations(patches[0].get_img(), label_map, bbox_map)
        # annotated_img = np.array(patches[0].get_img())
        # for ann in annotations:
        #     x, y, minx, miny, maxx, maxy = ann
        #     cv2.rectangle(annotated_img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        # cv2.imwrite('/data/ann.png', annotated_img)
        # print len(patches[0].annotations)
