import abc
import matplotlib.pyplot as plt

import numpy as np

from detection.dataset.fcn_mask_generator import FCNMaskGenerator
from detection.detectors.base import AbstractDetector
from detection.detectors.model_optimization import start_training
from detection.utils.image_utils import local_maxima, get_annotated_img
from detection.utils.logger import logger
from detection.utils.metric_utils import get_matches


class FCNDetector(AbstractDetector):
    '''
    Fully convolutional detecter. All detectors of such type should have same input and output size.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.patch_size = input_shape[1], input_shape[2]
        self.input_channels = input_shape[3]
        self.out_channels = no_classes
        self.output_shape = [self.batch_size, input_shape[1], input_shape[2], self.out_channels]
        self.learning_rate = learning_rate
        self.no_classes = no_classes
        self.weight_file = weight_file
        self.model = self.build_model()

    @abc.abstractmethod
    def build_model(self):
        '''
        Should be implemented by each detector. Only contract is that the resulting model should accept
        self.input_shape and self.output_shape
        :return: A keras model.
        '''
        pass

    def _get_data_generator(self, dataset, testing_ratio, validation_ratio):
        '''
        Instantiate appropriate data generator.
        '''
        return FCNMaskGenerator(dataset, testing_ratio, validation_ratio)

    def predict_patch(self, input_img):
        '''
        Predicts the output for input image.
        :param input_img: Input image compatible with model input shape.
        :return: Predicted response map.
        '''
        model_input = np.expand_dims(input_img, axis=0)
        model_output = self.model.predict(model_input)
        response_map = np.squeeze(model_output, axis=3)[0]
        return response_map

    def predict_complete(self, image):
        '''
        Predicts the response map for input of any size greater than model input shape. It devides image into
        multiple patches and takes maxima in overlapping regions.
        :param image: Input image of any shape
        :return: Full response map of image.
        '''
        patch_size = self.patch_size
        step_size = (150, 150)
        image_shape = image.shape
        response_map = np.zeros([image_shape[0], image_shape[1]])

        x = range(0, image_shape[0] - patch_size[0], step_size[0])
        y = range(0, image_shape[1] - patch_size[1], step_size[1])
        x.append(image_shape[0] - patch_size[0])
        y.append(image_shape[1] - patch_size[1])
        xy = [(i, j) for i in x for j in y]
        for i, j in xy:
            img_patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            img_patch = (img_patch - img_patch.mean()) / (img_patch.std() + 1e-9)
            out_map = self.predict_patch(img_patch)
            response_map[i:i + patch_size[0], j:j + patch_size[1]] = np.maximum(
                response_map[i:i + patch_size[0], j:j + patch_size[1]], out_map)

        return response_map

    def get_predictions(self, dataset):
        frames = dataset.all_frames
        total_predictions = total_annotations = total_matches = 0
        # for idx in [104,47,84,61,135,212,28,120,196,180,48,162,73,112,60,202,32,175
        #         ,216,204,223,36,132,117,239,75,34,111,193,89,203,123,178,127,35,248
        #         ,87,158,110,194,91,187,145,51,16,58,21,49,64,68]:
        #     frame = frames[idx]
        for frame in frames:
            response_map = self.predict_complete(frame.img_data)
            actual_annotations = np.array([(ann[0], ann[1]) for ann in frame.annotations])
            predicted_annotations = local_maxima(response_map, 20, 0.4)
            matches = get_matches(predicted_annotations, actual_annotations)
            logger.info('Frame:{} Predicted:{}, actual:{}, matches:{}', frame.img_id, len(predicted_annotations),
                        len(frame.annotations),
                        len(matches))
            total_annotations += len(frame.annotations)
            total_matches += len(matches)
            total_predictions += len(predicted_annotations)
            false_pos = set(range(len(predicted_annotations))) - {match[0] for match in matches}
            unmatched = set(range(len(actual_annotations))) - {match[1] for match in matches}
            unmatched_annotations = [frame.annotations[idx] for idx in unmatched]
            print false_pos
            annotated_img = np.array(response_map)
            bbox_size = (20, 20)
            if len(false_pos) > 1:
                # for ann in unmatched_annotations:
                #     x, y, s = ann
                #     cv2.rectangle(annotated_img, (x - bbox_size[0], y - bbox_size[1]),
                #                   (x + bbox_size[0], y + bbox_size[1]),
                #                   1, 2)
                fp_annotations = [map(int, predicted_annotations[fp]) for fp in false_pos]
                plt.figure(1), plt.imshow(response_map)
                plt.figure(2), plt.imshow(frame.annotated_img())
                plt.figure(3), plt.imshow(get_annotated_img(frame.img_data, fp_annotations, (15, 15)))
                plt.show()
        logger.info('Total matches:{}, predictions:{}, actual:{}', total_matches, total_predictions, total_annotations)
        precision = total_matches * 1.0 / total_annotations
        recall = total_matches * 1.0 / total_predictions
        f1 = 2 * precision * recall / (precision + recall)
        logger.info('Precision:{}, recall:{}, F1:{}', precision, recall, f1)

    def train(self, dataset, batch_size, checkpoint_dir, samples_per_epoc, nb_epocs, testing_ratio, validation_ratio,
              nb_validation_samples):
        '''
        Starts training of the model with data provided by dataset.
        '''
        generator = self._get_data_generator(dataset, testing_ratio, validation_ratio)
        dataset_generator = generator.fcn_data_generator(batch_size, self.patch_size, self.no_classes)
        validation_generator = generator.fcn_data_generator(batch_size, self.patch_size, self.no_classes,
                                                            dataset='validation')
        start_training(checkpoint_dir, self.model, dataset_generator, samples_per_epoc, nb_epocs,
                       callbacks=[], validation_generator=validation_generator,
                       nb_val_samples=nb_validation_samples)
