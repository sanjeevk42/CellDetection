from keras.engine import Model
from keras.layers import Convolution2D, Dense, Reshape, Flatten
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.model_optimization import start_training
from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class Detectnet(object):
    '''
    A variant of detectnet architecture.
    https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/.
    '''

    def __init__(self, base_model_fn, last_layer_name):
        self.base_model_fn = base_model_fn
        self.last_layer_name = last_layer_name
        self.base_model = self.base_model_fn(include_top=True)

    def fully_conv(self):
        '''
        The top layers after base model are fully convolutional.
        '''
        model = Model(input=self.base_model.input, output=self.base_model.get_layer(self.last_layer_name).output)
        # 7*7*1 response map
        class_out = Convolution2D(1, 3, 3, name='class_out', border_mode='same', activation='sigmoid')(model.output)
        # 7*7*4 response map
        bb_out = Convolution2D(4, 3, 3, name='bb_out', border_mode='same', activation='linear')(model.output)

        model = Model(self.base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_squared_error'})
        logger.info('Compiled fully conv with output:{}', model.output)
        return model

    def fully_connected(self):
        '''
        The top layers after base model are fully connected.
        '''
        model = Model(input=self.base_model.input, output=self.base_model.get_layer(self.last_layer_name).output)
        model = Convolution2D(28 * 28 * 1, 7, 7, activation='relu')(model.output)
        model = Flatten()(model)
        class_out = Dense(28 * 28 * 1, name='fc_class', activation='sigmoid')(model)
        class_out = Reshape((28, 28, 1), name='class_out')(class_out)

        bb_out = Dense(28 * 28 * 4, name='fc_bb', activation='linear')(model)
        bb_out = Reshape((28, 28, 4), name='bb_out')(bb_out)

        model = Model(self.base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_squared_error'})
        logger.info('Compiled fc with output:{}', model.output)
        return model


if __name__ == '__main__':
    dataset_dir = '/data/lrz/hm-cell-tracking/sequences_A549/annotations'
    checkpoint_dir = '/data/training/detectnet'
    batch_size = 10
    samples_per_epoc = 6000
    nb_epocs = 500
    patch_size = (224, 224)
    grid_size = (32, 32)

    detector = Detectnet(ResNet50, 'activation_43')
    model = detector.fully_conv()
    image_dataset = ImageDataset(dataset_dir)
    dataset_generator = image_dataset.grid_dataset_generator(batch_size, patch_size, grid_size)
    start_training(dataset_dir, checkpoint_dir, model, dataset_generator, samples_per_epoc, nb_epocs)
