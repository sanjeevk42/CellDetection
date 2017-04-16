import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.ndimage import interpolation

from detection.dataset.dataset_generator import DatasetGenerator
from detection.dataset.image_dataset import ImageDataset
from detection.utils import image_utils
from detection.utils.image_utils import get_bbox
from detection.utils.logger import logger


class GridDatasetGenerator(DatasetGenerator):
    '''
    Data set generator for network architectures which predict grid outputs such as Yolo and detectnet.
    '''

    def __init__(self, dataset, test_ratio, validation_ratio):
        super(GridDatasetGenerator, self).__init__(dataset, test_ratio, validation_ratio)

    def grid_dataset_generator(self, batch_size, patch_size=(224, 224), grid_size=(28, 28), dataset='training',
                               sampling_type='random'):
        '''
        Training data generator for grid based predictions.
        '''
        all_patches = []
        sample_idx = self.dataset_idx(dataset)
        for idx in sample_idx:
            frame = self.all_frames[idx]
            if sampling_type == 'random':
                all_patches.extend(frame.get_random_patches(patch_size, 1))
            else:
                all_patches.extend(frame.sequential_patches(patch_size, (200, 200)))
        logger.info('Total patches:{}', len(all_patches))
        while True:
            # Randomly select patches from sequential patches ...
            patches_idx = np.random.randint(0, len(sample_idx), batch_size)
            class_batch = []
            bbout_batch = []
            input_batch = []
            for idx in patches_idx:
                frame = self.all_frames[sample_idx[idx]]
                # patch = frame.get_random_patches(patch_size, 1)[0]
                patch = frame.sequential_patches(patch_size, (200, 200))[0]
                label_map, bbox_map = self.grid_ground_truth(patch, grid_size)
                # plt.figure(1), plt.imshow(np.squeeze(label_map))
                # plt.figure(2), plt.imshow(patch.annotated_img())
                # plt.show()
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
        bbox_map = np.zeros(bbox_map_shape, dtype=np.int32)
        label_map = np.zeros(response_map_shape + [1])

        bbox_size = patch.frame_info.bbox
        for i in range(int(bbox_map_shape[0])):
            for j in range(int(bbox_map_shape[1])):
                grid_index = np.array([i, j]) * grid_size
                grid_bound = tuple(grid_index) + tuple(grid_index + grid_size)
                best_annotation_idx = -1
                max_intersection = 0
                for idx, ann in enumerate(patch.annotations):
                    y, x, s = ann
                    ann_bound = (x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1])
                    intersection = image_utils.intersection(grid_bound, ann_bound)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        best_annotation_idx = idx

                if best_annotation_idx > 0:
                    grid_center = grid_index + np.array(grid_size) / 2
                    y, x, s = patch.annotations[best_annotation_idx]
                    ann_bound = np.array((x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1]))
                    ann_bound_norm = ann_bound - np.concatenate([grid_center, grid_center])
                    label_map[i][j] = 1  # currenly only binary segmentation is being done ...
                    bbox_map[i][j] = ann_bound_norm

        return label_map, bbox_map

    def grid_patch_dataset_generator(self, batch_size, patch_size=(224, 224), grid_size=(28, 28), nb_objects=5,
                                     dataset='training'):
        '''
        Training data generator for grid patch based predictions where nb_objects labels and bounding boxes are
        predicted at each grid.
        '''
        all_patches = []
        sample_idx = self.dataset_idx(dataset)
        for idx in sample_idx:
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
                    key=lambda x: image_utils.intersection(grid_bounds, get_bbox(x, bbox_size)),
                    reverse=True)

                grid_ann_length = len(grid_annotations)
                for ann_idx in range(min(grid_ann_length, nb_objects)):
                    x, y, s = grid_annotations[ann_idx]
                    confidence = image_utils.intersection(grid_bounds, get_bbox(ann, bbox_size))
                    obj_center = np.array([x, y]) - grid_index
                    bbox_map[i][j][ann_idx] = np.concatenate([obj_center, bbox_size, [confidence]])
                    label_map[i][j][ann_idx] = 1

            return label_map, bbox_map


if __name__ == '__main__':
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations/', '.png')
    fcn_mask_gen = GridDatasetGenerator(dataset, 0, 0).grid_dataset_generator(1, (512, 512), (16, 16))
    for data in fcn_mask_gen:
        input, output = data
        input_img = np.squeeze(input['input_1'], axis=0)
        plt.figure(1), plt.imshow(input_img)
        class_score, bb_score = output['class_out'], output['bb_out']
        annotations = image_utils.feature_to_annotations(input_img, np.squeeze(class_score), np.squeeze(bb_score))
        ann_img = image_utils.draw_bboxes(input_img, annotations)
        plt.figure(2), plt.imshow(ann_img)
        plt.figure(3), plt.imshow(np.squeeze(class_score))
        zoom_class_score = interpolation.zoom(np.squeeze(image_utils.normalize(class_score)), (16, 16))

        zoom_class_score = np.repeat(np.expand_dims(zoom_class_score, axis=-1).astype(np.float32), 3, axis=-1)
        c = np.array((1, 0, 0), dtype=np.float32)
        zoom_class_score = zoom_class_score * c.reshape(1,1,3)
        out1 = cv2.addWeighted(input_img, 0.5, zoom_class_score, 0.5, 0)
        for i in range(32):
            out1 = cv2.line(out1, (16*i, 0), (16*i, 511), (0,0, 255), 1)
            out1 = cv2.line(out1, (0, 16*i), (511, 16 * i), (0, 0, 255), 1)
        cv2.imwrite('/home/sanjeev/overlay.jpg', out1[:300,:300])
        cv2.imwrite('/home/sanjeev/original.jpg', input_img[:300,:300])
        # plt.show()
        print(class_score, bb_score)
