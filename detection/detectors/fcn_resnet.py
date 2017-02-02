import matplotlib.pyplot as plt
import numpy as np

import cv2
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dropout, Deconvolution2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.fcn_detecter import FCNDetector
from detection.detectors.unet import UNet
from detection.models.resnet50 import ResNet50
from detection.utils.image_utils import get_annotated_img, local_maxima
from detection.utils.logger import logger


class FCNResnet50(FCNDetector):
    def __init__(self, input_shape, learning_rate, no_classes, weight_file):
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
        for layer in model.layers:
            layer.trainable = False
            if layer.name == last_layer_name:
                break

        if self.weight_file:
            logger.info('Loading weights from :{}', self.weight_file)
            model.load_weights(self.weight_file)

        logger.info('Compiled fully conv with output:{}', model.output)
        # model.summary()
        return model


def normalize(image, sigma_low=0, sigma_high=30):
    '''background subtraction.

    The background is estimated by a wide-width Gaussian kernel.
    Make sure the sigma is much bigger than the objects of interest,
    but still captures the scale at which background changes occur.

    Parameters
    ----------
    image : array_like
        image to be normalized.
    sigma_low : float, optional
        kernel width to remove noise.
    sigma_high : float, optional
        kernel width to estimate background.

    Returns
    -------
    normed : array_like
        normalized image.

    '''
    from scipy.ndimage import fourier_gaussian
    from numpy.fft import fft2, ifft2
    from numpy import real

    image = image.astype(float)
    fftimg = fft2(image)

    # low-pass.
    if sigma_low > 0:
        fftimg = fourier_gaussian(fftimg, sigma=sigma_low)

    # high-pass.
    fftimg -= fourier_gaussian(fftimg, sigma=sigma_high)

    # transform back.
    filtered = real(ifft2(fftimg))
    return filtered



if __name__ == '__main__':
    batch_size = 1
    weight_file = '../../weights/fcn_resnet.hdf5'
    detector = FCNResnet50([batch_size, 224, 224, 3], 1e-3, 1, weight_file)

    img = cv2.imread('/data/lrz/hm-cell-tracking/sequences_150602_3T3/sample_01/cam0_0154.jpg')
    response_map = detector.predict_complete(img)
    plt.imshow(response_map)
    plt.show()
    plt.savefig('/data/lrz/hm-cell-tracking/sequences_150602_3T3/predictions_fcn_01/rmap_cam0_0154.jpg')
    plt.close('all')
    predicted_annotations = local_maxima(response_map, 20, 0.4)
    ann_img = get_annotated_img(img, predicted_annotations, (15, 15))
    plt.imshow(ann_img)
    plt.savefig('/data/lrz/hm-cell-tracking/sequences_150602_3T3/predictions_fcn_01/cam0_0154.jpg')
    plt.close('all')

    # dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations/', '00_bw.png', normalize=False)
    # dataset = ImageDataset('/data/lrz/hm-cell-tracking/annotations/in', '.jpg', normalize=False)

    # training_args = {
    #     'dataset': dataset,
    #     'batch_size': batch_size,
    #     'checkpoint_dir': '/data/cell_detection/test',
    #     'samples_per_epoc': 4,
    #     'nb_epocs': 500,
    #     'testing_ratio': 0.2,
    #     'validation_ratio': 0.1,
    #     'nb_validation_samples': 6
    #
    # }
    # detector.train(**training_args)
    # detector.get_predictions(dataset, range(dataset.dataset_size), '/data/cell_detection/fcn_31_norm/predictions/')
    # image = np.array(cv2.imread('/data/lrz/hm-cell-tracking/sequences_150602_3T3/sample_01/cam0_0001.jpg'), dtype=np.float64)
    # image -= (53, 53, 53)
