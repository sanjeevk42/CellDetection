from keras.engine import Model
from keras.layers import Convolution2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.bbox_detecter import BBoxDetector
from detection.detectors.model_optimization import start_training
from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class YoloDetector(BBoxDetector):
    '''
    YOLO detector on resnet50.
    '''

    def __init__(self, input_shape, no_classes, grid_size, weight_file=None):
        super(YoloDetector, self).__init__(input_shape, no_classes, grid_size, weight_file)

    def build_model(self):
        '''
        The top layers after base model are fully connected.
        '''
        last_layer_name = 'activation_11'
        base_model = ResNet50(input_tensor=self.input_shape)
        base_model_out = base_model.get_layer(last_layer_name).output
        model = Model(input=base_model.input, output=base_model_out)
        class_out = Convolution2D(self.no_classes, 1, 1, border_mode='same', activation='sigmoid')(model)

        bb_out = Convolution2D(4, 1, 1, border_mode='same')(model)

        model = Model(base_model.input, output=[class_out, bb_out])

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
    model = detector.build_model(nb_objects)

    dataset = ImageDataset(dataset_dir)
    dataset_generator = dataset.grid_patch_dataset_generator(batch_size, patch_size, grid_size=grid_size,
                                                             nb_objects=nb_objects)
    start_training(dataset_dir, checkpoint_dir, model, dataset_generator, samples_per_epoc, nb_epocs)
