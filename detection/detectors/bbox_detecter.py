import abc

from detection.detectors.model_optimization import start_training


class BBoxDetecter(object):
    '''
    Paarent class for all bounding box type detectors.
    '''

    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape, learning_rate, no_classes, grid_size, weight_file=None):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.no_classes = no_classes
        self.grid_size = grid_size
        self.weight_file = weight_file
        self.model = self._build_model()

    @abc.abstractmethod
    def _build_model(self):
        '''
        Should be implemented by each detector. Only contract is that the resulting model should accept
        self.input_shape and self.output_shape
        :return: A keras model.
        '''
        pass

    @abc.abstractmethod
    def _get_data_generator(self, dataset, testing_ratio, validation_ratio):
        '''
        Instantiate appropriate data generator.
        '''
        pass

    def train(self, dataset, batch_size, checkpoint_dir, samples_per_epoc, nb_epocs, testing_ratio, validation_ratio,
              nb_validation_samples):
        '''
        Starts training of the model with data provided by dataset.
        '''
        generator = self._get_data_generator(dataset, testing_ratio, validation_ratio)
        dataset_generator = generator.grid_dataset_generator(batch_size, self.patch_size, self.no_classes)
        validation_generator = generator.grid_dataset_generator(batch_size, self.patch_size, self.no_classes,
                                                            dataset='validation')
        start_training(checkpoint_dir, self.model, dataset_generator, samples_per_epoc, nb_epocs,
                       callbacks=[], validation_generator=validation_generator,
                       nb_val_samples=nb_validation_samples)