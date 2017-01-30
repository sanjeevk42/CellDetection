import matplotlib.pyplot as plt

import numpy as np

from detection.dataset.dataset_generator import DatasetGenerator
from detection.dataset.image_dataset import ImageDataset


class FCNMaskGenerator(DatasetGenerator):
    '''
    Data set generator for generating training and test data for fully connected type networks.
    '''

    def __init__(self, dataset, test_ratio, validation_ratio, horizontal_flip=False):
        super(FCNMaskGenerator, self).__init__(dataset, test_ratio, validation_ratio)

    def fcn_data_generator(self, batch_size, patch_size, no_classes, dataset='training', sampling_type='random'):
        while True:
            image_batch = []
            response_maps = []
            idx = self.dataset_idx(dataset)
            while True:
                frame_id = np.random.randint(0, len(idx))
                frame = self.dataset.all_frames[frame_id]
                img_patches = frame.get_random_patches(patch_size, 1)
                for patch in img_patches:
                    img = patch.get_img()
                    img = (img - img.mean()) / (img.std() + 1e-9)
                    image_batch.append(img)
                # image_batch.extend([img_patch.get_img() for img_patch in img_patches])
                img_masks = [img_patch.ann_mask(no_classes) for img_patch in img_patches]
                response_maps.extend(img_masks)
                if len(image_batch) == batch_size:
                    break

            image_batch = np.array(image_batch)
            # logger.info('image batch shape:{}, dataset:{}, batch_size:{}', image_batch.shape, dataset, batch_size)
            yield ({'input_1': np.array(image_batch)}, {'class_out': np.array(response_maps)})


if __name__ == '__main__':
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations', '.png')
    fcn_mask_gen = FCNMaskGenerator(dataset, 0.2, 0.1).fcn_data_generator(1, (224, 224), 1)
    for data in fcn_mask_gen:
        input, output = data
        input_img = input['input_1']
        plt.figure(1), plt.imshow(input_img.squeeze(axis=0))
        out_img = output['class_out']
        plt.figure(2), plt.imshow(np.squeeze(out_img))
        plt.show()
