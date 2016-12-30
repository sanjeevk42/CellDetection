from keras.engine import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.model_optimization import start_training
from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class YoloDetector(object):
    '''
    YOLO detector on resnet50.
    '''

    def __init__(self, base_model_fn, last_layer_name):
        self.base_model_fn = base_model_fn
        self.last_layer_name = last_layer_name
        self.base_model = self.base_model_fn(include_top=True)

    def fully_connected(self, nb_objects):
        '''
        The top layers after base model are fully connected.
        '''
        model = Model(input=self.base_model.input, output=self.base_model.get_layer(self.last_layer_name).output)
        model = Flatten()(model.output)
        model = Dense(256, activation='relu')(model)
        model = Dropout(.7)(model)

        model = Dense(2048, activation='relu')(model)

        class_out = Dense(7 * 7 * nb_objects, name='fc_class', activation='sigmoid')(model)
        class_out = Reshape((7, 7, nb_objects), name='class_out')(class_out)

        bb_out = Dense(7 * 7 * 5 * nb_objects, name='fc_bb', activation='linear')(model)
        bb_out = Reshape((7, 7, nb_objects, 5), name='bb_out')(bb_out)

        model = Model(self.base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_squared_error'})
        logger.info('Compiled fc with output:{}', model.output)
        return model


if __name__ == '__main__':
    dataset_dir = '/data/lrz/hm-cell-tracking/sequences_A549/annotations'
    checkpoint_dir = '/data/training/yolo'
    nb_objects = 5
    batch_size = 100
    samples_per_epoc = 5000
    nb_epocs = 500
    patch_size = (224, 224)
    grid_size = (32, 32)

    detector = YoloDetector(ResNet50, 'activation_48')
    model = detector.fully_connected(nb_objects)

    dataset = ImageDataset(dataset_dir)
    dataset_generator = dataset.grid_patch_dataset_generator(batch_size, patch_size, grid_size=grid_size,
                                                             nb_objects=nb_objects)
    start_training(dataset_dir, checkpoint_dir, model, dataset_generator, samples_per_epoc, nb_epocs)
