import glob
import os

import click
import cv2
import numpy as np

from detection.detectors.fcn_resnet import FCNResnet50
from detection.detectors.unet import UNet
from detection.utils.logger import logger


def normalize(img):
    min_i = np.min(img)
    max_i = np.max(img)
    img = (img - min_i) / (max_i - min_i)
    img = img * 255
    return img


@click.command()
@click.argument("model", 'Name of model to use. e.g. resnet or unet')
@click.argument("input_dir", 'Input directory path', type=click.Path(exists=True))
@click.argument("out_dir", 'Output directory path.', type=click.Path(exists=True))
@click.argument("file_pattern", 'Pattern of files to convert. e.g. *jpg')
def predict_all(model, input_dir, out_dir, file_pattern):
    batch_size = 1
    if model == 'resnet':
        detector = FCNResnet50([batch_size, 224, 224, 3], 1e-3, 1, weight_file='weights/fcn_resnet.hdf5')
        step_size = (160, 160)
    elif model == 'unet':
        detector = UNet([batch_size, 252, 252, 3], 1e-3, 1, weight_file='weights/unet.hdf5')
        step_size = (180, 180)
    all_files = glob.glob(input_dir + file_pattern)
    for fn in all_files:
        logger.info('Processing file:{}', fn)
        base_filename = os.path.basename(fn)
        image = cv2.imread(fn)
        response_map = detector.predict_complete(image, step_size)
        rm_norm = normalize(response_map)
        out_path = out_dir + 'response_map_{}'.format(base_filename)
        logger.info('Writing response map at :{}', out_path)
        cv2.imwrite(out_path, rm_norm)


if __name__ == '__main__':
    predict_all()
    # predict_all('resnet', '/data/lrz/hm-cell-tracking/sequences_150602_3T3/sample_01/',
    #             '/data/lrz/hm-cell-tracking/sequences_150602_3T3/predictions_01/', '*.jpg')
