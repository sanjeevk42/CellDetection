from keras.engine import Input
import os
from detection.detectors.fcn_detecter import FCNDetector
from detection.models.vgg16 import VGG16
from keras.engine import Model
from keras.layers import Convolution2D, Dropout, Deconvolution2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.utils.logger import logger
import cv2
import matplotlib.pyplot as plt

class FCNVGG16(FCNDetector):
    '''

    '''

    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        super(FCNVGG16, self).__init__(input_shape, learning_rate, no_classes, weight_file)

    def build_model(self):
        input = Input(batch_shape=self.input_shape, name='input_1')
        base_model = VGG16(input_tensor=input)
        last_layer_name = 'block3_pool'
        base_model_out = base_model.get_layer(last_layer_name).output

        model = Model(input=base_model.input, output=base_model_out)
        no_features = base_model_out.get_shape()[3].value
        model = Convolution2D(no_features, 1, 1, border_mode='same', activation='relu')(model.output)
        model = Dropout(0.5)(model)
        class_out = Deconvolution2D(self.out_channels, 8, 8, output_shape=self.output_shape, subsample=(8, 8),
                                    activation='sigmoid', name='class_out')(model)

        model = Model(base_model.input, output=class_out)
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy'}, metrics=['binary_accuracy'])
        for layer in model.layers:
            layer.trainable = False
            if layer.name == last_layer_name:
                break

        if self.weight_file:
            logger.info('Loading weights from :{}', self.weight_file)
            model.load_weights(weight_file)

        model.summary()
        return model

def start_training():
    batch_size = 8
    detector = FCNVGG16([batch_size, 224, 224, 3], 1e-3, 1)
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
    weight_file = None#'/data/cell_detection/fcn_vgg/model_checkpoints/model.hdf5'
    detector = FCNVGG16([batch_size, 224, 224, 3], 1e-3, 1, weight_file)
    dataset = ImageDataset('/data/lrz/hm-cell-tracking/annotations/in', '.jpg', normalize=False)
    detector.get_predictions(dataset, range(dataset.dataset_size), '/data/cell_detection/fcn_31_norm/predictions/')

if __name__ == '__main__':
    # calculate_score()
    batch_size = 1
    weight_file = '/data/cell_detection/old/fcn_vgg/model_checkpoints/model.hdf5'
    detector = FCNVGG16([batch_size, 224, 224, 3], 1e-3, 1, weight_file)

    img = cv2.imread('/data/lrz/hm-cell-tracking/sequences_A549/annotations/00362_bw.png')
    response_map = detector.predict_complete(img)
    plt.figure(1), plt.imshow(response_map)
    plt.figure(2), plt.imshow(img)
    plt.show()
    # start_training()