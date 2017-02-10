import matplotlib.pyplot as plt

import cv2
from keras.engine import Input
from keras.engine import Model
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Reshape
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Deconvolution2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.fcn_detecter import FCNDetector
from detection.models.resnet50 import ResNet50
from detection.utils.image_utils import get_annotated_img, local_maxima
from detection.utils.logger import logger

class FCNSimpleCNN(FCNDetector):
    '''
    Simple CNN with 5 convolutional layers.
    '''
    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        super(FCNSimpleCNN, self).__init__(input_shape, learning_rate, no_classes, weight_file)

    def build_model(self):
        input = Input(batch_shape=self.input_shape)
        x = ZeroPadding2D((3, 3))(input)
        x = Convolution2D(16, 5, 5, border_mode='same', name='conv1')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool1')(x)

        x = Convolution2D(32, 5, 5, name='conv2', border_mode='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool2')(x)
        x = Dropout(0.6)(x)

        x = Convolution2D(64, 5, 5, border_mode='same', name='conv3')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool3')(x)
        x = Dropout(0.6)(x)

        x = Convolution2D(64, 5, 5, border_mode='same', name='conv4')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool4')(x)
        x = Dropout(0.6)(x)

        x = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='conv5')(x)
        x = Dropout(0.5)(x)
        class_out = Deconvolution2D(1, 8, 8, output_shape=self.output_shape, subsample=(8, 8),
                                    activation='sigmoid', name='class_out')(x)

        model = Model(input, output=class_out)

        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy'}, metrics=['binary_accuracy'])
        model.summary()
        return model

def start_training():
    batch_size = 8
    detector = FCNSimpleCNN([batch_size, 224, 224, 3], 1e-3, 1)
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations/', '.png', normalize=False)

    training_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'checkpoint_dir': '/data/cell_detection/test',
        'samples_per_epoc': 4000,
        'nb_epocs': 500,
        'testing_ratio': 0.2,
        'validation_ratio': 0.1,
        'nb_validation_samples': 400

    }
    detector.train(**training_args)


def calculate_score():
    batch_size = 1
    weight_file = '/data/cell_detection/resnet_random/model_checkpoints/model.hdf5'
    detector = FCNSimpleCNN([batch_size, 224, 224, 3], 1e-3, 1, weight_file)
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/annotations/in', '.jpg', normalize=False)
    detector.get_predictions(dataset, range(dataset.dataset_size), '/data/cell_detection/fcn_31_norm/predictions/')


if __name__ == '__main__':
    start_training()
    calculate_score()