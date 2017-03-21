from fire import Fire
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dropout, Deconvolution2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.fcn_detecter import FCNDetector
from detection.models.resnet50 import ResNet50
from detection.utils.logger import logger


class FCNResnet50(FCNDetector):
    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        super(FCNResnet50, self).__init__(input_shape, learning_rate, no_classes, weight_file)

    def build_model(self):
        input_tensor = Input(batch_shape=self.input_shape)
        last_layer_name = 'activation_22'
        base_model = ResNet50(include_top=False, input_tensor=input_tensor, weights=None)
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
        for i, layer in enumerate(model.layers):
            layer.trainable = False
            if layer.name == last_layer_name:
                break

        if self.weight_file:
            logger.info('Loading weights from :{}', self.weight_file)
            model.load_weights(self.weight_file)

        logger.info('Compiled fully conv with output:{}', model.output)
        model.summary()
        return model


def start_training(batch_size, checkpoint_dir, dataset_dir, samples_per_epoc, nb_epocs, validation_samples,
                   file_ext='.png', weight_file=None):
    """
    Usage:python -m detection.detectors.fcn_resnet start-training 1 '/data/cell_detection/test' \
         '/data/lrz/hm-cell-tracking/sequences_A549/annotations/' 1 10 1
    """
    detector = FCNResnet50([batch_size, 224, 224, 3], 1e-3, 1, weight_file=weight_file)
    dataset = ImageDataset(dataset_dir, file_ext, normalize=False)

    training_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'checkpoint_dir': checkpoint_dir,
        'samples_per_epoc': samples_per_epoc,
        'nb_epocs': nb_epocs,
        'testing_ratio': 0.2,
        'validation_ratio': 0.1,
        'nb_validation_samples': validation_samples

    }
    detector.train(**training_args)


def evaluate_model(dataset_dir, weight_file, out_dir, file_ext='.png'):
    """
    Usage: python -m detection.detectors.fcn_resnet evaluate-model '/data/lrz/hm-cell-tracking/annotations/in/'  \
       '/data/cell_detection/resnet23_imagenet/model_checkpoints/model.hdf5' '/data/training/detectnet/' --file-ext '.jpg'
    """
    batch_size = 1
    detector = FCNResnet50([batch_size, 224, 224, 3], 1e-3, 1, weight_file)
    dataset = ImageDataset(dataset_dir, file_ext, normalize=False)
    detector.get_predictions(dataset, range(len(dataset.all_frames)), out_dir)


if __name__ == '__main__':
    Fire({'evaluate-model': evaluate_model, 'start-training': start_training})
