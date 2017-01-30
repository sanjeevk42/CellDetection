import os

from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dense, Reshape, Flatten
from keras.optimizers import Adam

from detection.dataset.grid_dataset_generator import GridDatasetGenerator
from detection.dataset.image_dataset import ImageDataset
from detection.detectors.bbox_detecter import BBoxDetecter
from detection.detectors.model_optimization import start_training
from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class Detectnet(BBoxDetecter):
    '''
    A variant of detectnet architecture.
    https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/.
    '''

    def __init__(self, input_shape, learning_rate, no_classes, grid_size, weight_file=None):
        super(Detectnet, self).__init__(input_shape, learning_rate, no_classes, grid_size, weight_file)

    def build_model(self):
        '''
        The top layers after base model are fully convolutional.
        '''
        last_layer_name = 'activation_49'
        input_tensor = Input(batch_shape=self.input_shape)
        base_model = ResNet50(include_top=False, input_tensor=input_tensor)
        model = Model(input=base_model.input, output=base_model.get_layer(last_layer_name).output)
        # 7*7*1 response map
        class_out = Convolution2D(1, 3, 3, name='class_out', border_mode='same', activation='sigmoid')(model.output)
        # 7*7*4 response map
        bb_out = Convolution2D(4, 3, 3, name='bb_out', border_mode='same', activation='linear')(model.output)

        model = Model(self.base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_squared_error'})
        if self.weight_file:
            model.load_weights(self.weight_file)
        logger.info('Compiled fully conv with output:{}', model.output)
        return model


    def _get_data_generator(self, dataset, testing_ratio, validation_ratio):
        '''
        Instantiate appropriate data generator.
        '''
        return GridDatasetGenerator(dataset, testing_ratio, validation_ratio)

if __name__ == '__main__':
    dataset_dir = '/data/lrz/hm-cell-tracking/sequences_A549/annotations'
    checkpoint_dir = '/data/training/detectnet'
    out_dir = os.path.join(checkpoint_dir, 'out')
    batch_size = 1
    samples_per_epoc = 4
    nb_epocs = 500
    nb_validation_samples = 2
    patch_size = (224, 224)
    grid_size = (32, 32)
    no_classes = 2

    detector = Detectnet(ResNet50, 'activation_22', batch_size)  # activation_48
    model = detector.fully_convolutional(no_classes)
    # model.load_weights('/data/cell_detection/fcn_deconv_seq1_norm/model_checkpoints/model.hdf5', by_name=False)
    model.summary()
    image_dataset = ImageDataset(dataset_dir)
    # output_writer = OutputWriter(image_dataset, out_dir, patch_size, (220, 220))
    dataset_generator = image_dataset.fcn_data_generator(batch_size, patch_size, no_classes)
    validation_generator = image_dataset.fcn_data_generator(batch_size, patch_size, no_classes,
                                                            dataset='validation')
    start_training(dataset_dir, checkpoint_dir, model, dataset_generator, samples_per_epoc, nb_epocs,
                   callbacks=[], validation_generator=validation_generator,
                   nb_val_samples=nb_validation_samples)
