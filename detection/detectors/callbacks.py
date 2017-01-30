import os

import numpy as np
from keras.callbacks import Callback

# from detection.dataset.image_dataset import ImageDataset
# from detection.detectors.detectnet import Detectnet
# from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class OutputWriter(Callback):
    def __init__(self, image_dataset, outdir, patch_size, step_size):
        self.image_dataset = image_dataset
        self.outdir = outdir
        self.patch_size = patch_size
        self.step_size = step_size

    def on_epoch_end(self, epoch, logs={}):
        for idx in self.image_dataset.validation_idx[:1]:
            frame = self.image_dataset.all_frames[idx]
            patches = frame.sequential_patches(self.patch_size, self.step_size)
            for i, patch in enumerate(patches):
                input_img = np.expand_dims(patch.get_img(), axis=0)
                response_map = np.squeeze(self.model.predict(input_img))
                logger.info('Frame:{} response min:{}, max:{}', frame.img_id, np.min(response_map), np.max(response_map))
                np.save(os.path.join(self.outdir, 'response_map_{}_{}.npy'.format(frame.img_id, i)), response_map)

# if __name__ == '__main__':
#     dataset_dir = '/data/lrz/hm-cell-tracking/sequences_A549/annotations'
#     checkpoint_dir = '/data/training/detectnet'
#     out_dir = os.path.join(checkpoint_dir, 'out')
#     batch_size = 1
#     samples_per_epoc = 2
#     nb_epocs = 500
#     nb_validation_samples = 1
#     patch_size = (224, 224)
#     grid_size = (32, 32)
#     no_classes = 2
#     image_dataset = ImageDataset(dataset_dir)
#     detector = Detectnet(ResNet50, 'activation_49')  # activation_48
#     model = detector.fully_convolutional(no_classes)
#
#     callback = OutputWriter(image_dataset, out_dir, patch_size, (200,200))
#     callback.model = model
#     callback.on_epoch_end(None)