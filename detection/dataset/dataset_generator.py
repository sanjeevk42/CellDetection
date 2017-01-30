import numpy as np

from detection.utils.logger import logger


class DatasetGenerator(object):
    '''
    Base dataset generator. Each data set generator generates input and output for training and validation.
    There should be one type of dataset generator for one particular type of training. e.g. FCNMaskGenerator
    should provide data for training fully connected networks. Dataset generator should implement the logic of
    pulling data from different dataset sources and should also perform data augmentation.
    '''

    def __init__(self, dataset, test_ratio=0.20, validation_ratio=0.10):
        self.dataset = dataset
        self.dataset_size = self.dataset.dataset_size
        self.split_dataset(test_ratio, validation_ratio)
        self.all_frames = self.dataset.all_frames
        print self.testing_idx

    def split_dataset(self, test_ratio, validation_ratio):
        '''
        Splits the dataset into training, testing and validation sets.
        '''
        idx = range(self.dataset_size)
        idx = np.random.permutation(idx)
        training_data_size = int(self.dataset_size * (1 - test_ratio))
        validation_size = int(training_data_size * validation_ratio)
        self.validation_idx = idx[:validation_size]
        self.training_idx = idx[validation_size:training_data_size]
        self.testing_idx = idx[training_data_size:]

        logger.info('Total dataset size:{}, training:{}, validation:{}, test:{}', self.dataset_size,
                    len(self.training_idx), len(self.validation_idx), len(self.testing_idx))

    def dataset_idx(self, dataset):
        if dataset == 'testing':
            sample_idx = self.testing_idx
        elif dataset == 'validation':
            sample_idx = self.validation_idx
        else:
            sample_idx = self.training_idx
        return sample_idx
