import matplotlib.pyplot as plt

import numpy as np

from detection.dataset.dataset_generator import DatasetGenerator
from detection.dataset.image_dataset import ImageDataset
from detection.utils import image_utils
from detection.utils.image_utils import get_bbox
from detection.utils.logger import logger


class PatchGenerator(DatasetGenerator):
    '''
    Data set generator which predicts center pixel class probabilities.
    '''

    def __init__(self, dataset, test_ratio, validation_ratio):
        super(PatchGenerator, self).__init__(dataset, test_ratio, validation_ratio)

    def patch_dataset_generator(self, batch_size, patch_size=(64, 64), no_of_objects=5, dataset='training',
                                sampling_type='random'):
        '''
        Dataset generator for no_of_objects predictions in image patch..
        '''
        # Only random sampling of image patches is implemented
        sample_idx = self.dataset_idx(dataset)
        all_patches = []
        for idx in sample_idx:
            all_patches.extend(self.all_frames[idx].sequential_patches(patch_size, (50, 50)))
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
        bbox_size = patch.frame_info.bbox
        patch_bound = (0, 0) + patch.patch_size
        labels = np.zeros(no_of_objects)
        bboxes = np.zeros((no_of_objects, 5))
        patch_annotations = list(patch.annotations)
        patch_annotations.sort(key=lambda x: image_utils.intersection(patch_bound, get_bbox(x, bbox_size)),
                               reverse=True)
        no_of_annotations = len(patch_annotations)
        for i in range(min(no_of_objects, no_of_annotations)):
            ann_bound = get_bbox(patch_annotations[i], bbox_size)
            # normalize bounding box
            # ann_bound = np.array(ann_bound, dtype=np.float32) / (patch.patch_size + patch.patch_size)
            bbox_confidence = image_utils.intersection(patch_bound, ann_bound) / image_utils.area(ann_bound)
            # print ann_bound, bbox_confidence
            bboxes[i] = list(ann_bound) + [bbox_confidence]
            labels[i] = 1
        return labels, bboxes


if __name__ == '__main__':
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/annotations/in', '40.jpg')
    fcn_mask_gen = PatchGenerator(dataset, 0.2, 0.1).patch_dataset_generator(1, (224, 224), 5)
    for data in fcn_mask_gen:
        input, output = data
        input_img = input['input_1']
        plt.figure(1), plt.imshow(input_img.squeeze(axis=0))
        class_score, bb_score = output['class_out'], output['bb_out']
        print class_score, bb_score
        # plt.figure(2), plt.imshow(np.squeeze(out_img))
        plt.show()
