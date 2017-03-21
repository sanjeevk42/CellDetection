import numpy as np
from fire import Fire
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dropout
from keras.optimizers import Adam

from detection.dataset.grid_dataset_generator import GridDatasetGenerator
from detection.dataset.image_dataset import ImageDataset
from detection.detectors.bbox_detecter import BBoxDetector
from detection.models.resnet50 import ResNet50
from detection.utils import image_utils
from detection.utils import metric_utils
from detection.utils.logger import logger


class Detectnet(BBoxDetector):
    '''
    A variant of detectnet architecture.
    https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/.
    '''

    def __init__(self, input_shape, no_classes, grid_size, weight_file=None):
        super(Detectnet, self).__init__(input_shape, no_classes, grid_size, weight_file)

    def build_model(self):
        '''
        The top layers after base model are fully convolutional.
        '''
        input_tensor = Input(batch_shape=self.input_shape)
        last_layer_name = 'activation_23'
        base_model = ResNet50(input_tensor=input_tensor)
        base_model_out = base_model.get_layer(last_layer_name).output
        model = Model(input=base_model.input, output=base_model_out)
        model = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(model.output)
        model = Dropout(0.5)(model)
        # model = MaxPooling2D((2, 2))(model)
        class_out = Convolution2D(self.no_classes, 1, 1, border_mode='same', activation='sigmoid', name='class_out')(
            model)

        bb_out = Convolution2D(4, 1, 1, border_mode='same', name='bb_out')(model)

        model = Model(base_model.input, output=[class_out, bb_out])

        optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_absolute_error'})
        logger.info('Compiled fc with output:{}', model.output)
        # model.summary()
        return model

    def _get_data_generator(self, dataset, testing_ratio, validation_ratio):
        '''
        Instantiate appropriate data generator.
        '''
        return GridDatasetGenerator(dataset, testing_ratio, validation_ratio)

    def predict_complete(self, image, step_size=(200, 200)):
        '''
        Predicts the response map for input of any size greater than model input shape. It devides image into
        multiple patches and takes maxima in overlapping regions.
        :param image: Input image of any shape
        :return: Full response map of image.
        '''
        patch_size = self.model.input_shape[1:3]
        image_shape = image.shape

        x = range(0, image_shape[0] - patch_size[0], step_size[0])
        y = range(0, image_shape[1] - patch_size[1], step_size[1])
        x.append(image_shape[0] - patch_size[0])
        y.append(image_shape[1] - patch_size[1])
        xy = [(i, j) for i in x for j in y]
        all_annotations = []
        for i, j in xy:
            img_patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            model_input = np.expand_dims(img_patch, axis=0)
            class_score, bb_score = self.model.predict(model_input)
            annotations = image_utils.feature_to_annotations(img_patch, np.squeeze(class_score), np.squeeze(bb_score))
            annotations = [(ann[1] + j, ann[0] + i, ann[3] + j, ann[2] + i) for ann in annotations]
            all_annotations.extend(annotations)
        return all_annotations

    def evaluate_dataset(self, image_dataset, frame_ids):
        total_predictions = total_annotations = total_matches = 0
        for idx in frame_ids:
            frame = image_dataset.all_frames[idx]
            predicted_bboxes = self.predict_complete(frame.img_data)
            predicted_bboxes = image_utils.group_bboxes(predicted_bboxes)
            predicted_annotations = [image_utils.get_cell_center(r) for r in predicted_bboxes]
            gt_annotations = [(ann[0], ann[1]) for ann in frame.annotations]
            matches = metric_utils.get_matches(predicted_annotations, gt_annotations)
            total_matches += len(matches)
            total_predictions += len(predicted_annotations)
            total_annotations += len(gt_annotations)
            recall_f, precision_f, f1_f = metric_utils.score_detections(predicted_annotations, frame.annotations,
                                                                        matches)
            logger.info('Processed frame:{}, precision:{}, recall:{}, f1:{}', frame.img_id, precision_f, recall_f, f1_f)
            #         plt.imshow(image_utils.get_annotated_img(frame.img_data, predicted_annotations,(15,15)))
            #         plt.show()
        precision = total_matches * 1.0 / total_predictions
        recall = total_matches * 1.0 / total_annotations
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1


def start_training(batch_size, checkpoint_dir, dataset_dir, file_ext='.png', weight_file=None):
    """
    Usage: python -m detection.detectors.detectnet start-training 1 '/data/cell_detection/test' \
            '/data/lrz/hm-cell-tracking/sequences_A549/annotations/'
    """
    no_classes = 1
    grid_size = (16, 16)
    detector = Detectnet([batch_size, 224, 224, 3], no_classes, grid_size, weight_file)
    dataset = ImageDataset(dataset_dir, file_ext, normalize=False)

    training_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'checkpoint_dir': checkpoint_dir,
        'samples_per_epoc': 1,
        'nb_epocs': 500,
        'testing_ratio': 0.2,
        'validation_ratio': 0.1,
        'nb_validation_samples': 2

    }
    detector.train(**training_args)


def evaluate_model(dataset_dir, weight_file, file_ext='.png'):
    """
    Evaluates model using all images from dataset_dir.
    Usage: python -m detection.detectors.detectnet evaluate-model '/data/lrz/hm-cell-tracking/annotations/in/' \
    '/data/training/detectnet/model_checkpoints/model.hdf5' --file-ext '.jpg'
    :param dataset_dir:
    :param weight_file:
    :param file_ext:
    :return:
    """
    batch_size = 1
    no_classes = 1
    grid_size = (16, 16)
    detector = Detectnet([batch_size, 224, 224, 3], no_classes, grid_size, weight_file)

    dataset = ImageDataset(dataset_dir, file_ext, normalize=False)
    precision, recall, f1 = detector.evaluate_dataset(dataset, range(len(dataset.all_frames)))
    logger.info("Precision:{}, recall:{}, f1:{}", precision, recall, f1)


if __name__ == '__main__':
    Fire({'evaluate-model': evaluate_model, 'start-training': start_training})
