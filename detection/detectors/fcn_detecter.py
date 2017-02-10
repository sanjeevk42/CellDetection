import abc
import csv
import os

import cv2
import numpy as np

from detection.dataset.fcn_mask_generator import FCNMaskGenerator
from detection.detectors.base import AbstractDetector
from detection.detectors.model_optimization import start_training
from detection.utils.image_utils import local_maxima, normalize
from detection.utils.logger import logger
from detection.utils.metric_utils import get_matches, score_detections


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
        norm_img = (input_img - input_img.mean()) / (input_img.std() + 1e-9)
        model_input = np.expand_dims(norm_img, axis=0)
        model_output = self.model.predict(model_input)
        response_map = np.squeeze(model_output, axis=3)[0]
        return response_map

    def predict_complete(self, image, step_size=(150, 150)):
        '''
        Predicts the response map for input of any size greater than model input shape. It devides image into
        multiple patches and takes maxima in overlapping regions.
        :param image: Input image of any shape
        :return: Full response map of image.
        '''
        patch_size = self.patch_size
        image_shape = image.shape
        response_map = np.zeros([image_shape[0], image_shape[1]])

        x = range(0, image_shape[0] - patch_size[0], step_size[0])
        y = range(0, image_shape[1] - patch_size[1], step_size[1])
        x.append(image_shape[0] - patch_size[0])
        y.append(image_shape[1] - patch_size[1])
        xy = [(i, j) for i in x for j in y]
        for i, j in xy:
            img_patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            out_map = self.predict_patch(img_patch)
            response_map[i:i + patch_size[0], j:j + patch_size[1]] = np.maximum(
                response_map[i:i + patch_size[0], j:j + patch_size[1]], out_map)

        return response_map

    def get_predictions(self, dataset, frame_idx, base_dir):
        total_predictions = total_annotations = total_matches = 0
        frame_metrics = []
        all_predictions = []
        all_fp = []
        all_fn = []
        for idx in frame_idx:
            frame = dataset.all_frames[idx]
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
            false_neg = set(range(len(actual_annotations))) - {match[1] for match in matches}
            false_neg_ann = [frame.annotations[idx] for idx in false_neg]
            false_pos_ann = [predicted_annotations[idx] for idx in false_pos]
            all_predictions.extend([[frame.img_id] + ann.tolist() for ann in predicted_annotations])
            all_fp.extend([[frame.img_id] + ann.tolist() for ann in false_pos_ann])
            all_fn.extend([[frame.img_id] + list(ann) for ann in false_neg_ann])
            rm_norm = normalize(response_map)
            cv2.imwrite(os.path.join(base_dir, 'response_map_{}'.format(frame.img_id)), rm_norm)
            # predicted_img = get_annotated_img(frame.img_data, predicted_annotations, (15, 15))
            # false_neg_img = get_annotated_img(frame.img_data, false_neg_ann, (15, 15))
            # fals_pos_img = get_annotated_img(frame.img_data, false_pos_ann, (15, 15))
            # cv2.imwrite(os.path.join(base_dir, 'fn_{}'.format(frame.img_id)), false_neg_img)
            # cv2.imwrite(os.path.join(base_dir, 'fp_{}'.format(frame.img_id)), fals_pos_img)
            # cv2.imwrite(os.path.join(base_dir, 'prediction_{}'.format(frame.img_id)), predicted_img)
            recall_f, precision_f, f1_f = score_detections(predicted_annotations, frame.annotations, matches)
            # img_id, total_ann, total_predictions, total_matches, recall, precision, f1
            frame_metric = [frame.img_id, len(frame.annotations), len(predicted_annotations),
                            len(matches), recall_f, precision_f, f1_f]
            frame_metrics.append(frame_metric)
        write_to_file(all_predictions, os.path.join(base_dir, 'predictions.csv'))
        write_to_file(all_fp, os.path.join(base_dir, 'fp.csv'))
        write_to_file(all_fn, os.path.join(base_dir, 'fn.csv'))
        write_to_file(frame_metrics, os.path.join(base_dir, 'score.csv'))
        logger.info('Total matches:{}, predictions:{}, actual:{}', total_matches, total_predictions, total_annotations)
        precision = total_matches * 1.0 / total_predictions
        recall = total_matches * 1.0 / total_annotations
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


def write_to_file(data, filename):
    with open(filename, 'w') as fw:
        csv_writer = csv.writer(fw)
        # csv_writer.writerow(headers)
        for row in data:
            csv_writer.writerow(row)
