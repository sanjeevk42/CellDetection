import matplotlib.pyplot as plt
import os

import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dropout
from keras.optimizers import Adam

from detection.dataset.grid_dataset_generator import GridDatasetGenerator
from detection.dataset.image_dataset import ImageDataset
from detection.detectors.bbox_detecter import BBoxDetector
from detection.models.resnet50 import ResNet50
from detection.utils import image_utils
from detection.utils.logger import logger


class Detectnet(BBoxDetector):
    '''
    A variant of detectnet architecture.
    https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/.
    '''

    def __init__(self, input_shape, no_classes, grid_size, weight_file=None):
        super(Detectnet, self).__init__(input_shape, no_classes, grid_size, weight_file=None)

    def build_model(self):
        '''
        The top layers after base model are fully convolutional.
        '''
        input_tensor = Input(batch_shape=self.input_shape)
        last_layer_name = 'activation_23'
        base_model = ResNet50(input_tensor=input_tensor)
        base_model_out = base_model.get_layer(last_layer_name).output
        model = Model(input=base_model.input, output=base_model_out)
        model = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(model.output)
        model = Dropout(0.5)(model)
        # model = MaxPooling2D((2, 2))(model)
        class_out = Convolution2D(self.no_classes, 1, 1, border_mode='same', activation='sigmoid', name='class_out')(
            model)

        bb_out = Convolution2D(4, 1, 1, border_mode='same', name='bb_out')(model)

        model = Model(base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_absolute_error'})
        logger.info('Compiled fc with output:{}', model.output)
        model.summary()
        return model

    def _get_data_generator(self, dataset, testing_ratio, validation_ratio):
        '''
        Instantiate appropriate data generator.
        '''
        return GridDatasetGenerator(dataset, testing_ratio, validation_ratio)


if __name__ == '__main__':
    dataset_dir = '/data/lrz/hm-cell-tracking/annotations/in'
    checkpoint_dir = '/data/training/detectnet'
    out_dir = os.path.join(checkpoint_dir, 'out')
    batch_size = 1
    samples_per_epoc = 1
    nb_epocs = 500
    nb_validation_samples = 1
    patch_size = (224, 224)
    grid_size = (16, 16)
    no_classes = 1
    weight_file = '/data/training/detectnet/model_checkpoints/model.hdf5'
    detector = Detectnet([batch_size, 224, 224, 3], no_classes, grid_size, weight_file)  # activation_48
    image_dataset = ImageDataset(dataset_dir, 'cam0_0001.jpg', normalize=False)
    detector.train(image_dataset, batch_size, checkpoint_dir, samples_per_epoc, nb_epocs, 0, 0.1,
                   nb_validation_samples)
    # model.load_weights('/data/cell_detection/fcn_deconv_seq1_norm/model_checkpoints/model.hdf5', by_name=False)

    # patch = image_dataset.all_frames[0].sequential_patches(patch_size, (200, 200))[0]
    # class_score, bb_score = detector.model.predict(np.expand_dims(patch.get_img(), axis=0))
    # class_score = np.squeeze(class_score)
    # bb_score = np.squeeze(bb_score)
    # annotations = image_utils.feature_to_annotations(patch.get_img(), np.squeeze(class_score), np.squeeze(bb_score))
    # ann_img = image_utils.draw_bboxes(patch.get_img(), annotations)
    # plt.figure(2), plt.imshow(ann_img)
    # plt.figure(3), plt.imshow(np.squeeze(class_score))
    # plt.show()
    # detector.train(image_dataset, batch_size, checkpoint_dir, samples_per_epoc, nb_epocs, 0, 0.5,
    #                nb_validation_samples)
    # output_writer = OutputWriter(image_dataset, out_dir, patch_size, (220, 220))
