from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Dropout
from keras.optimizers import Adam
from keras.engine import merge

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.fcn_detecter import FCNDetector


class UNet(FCNDetector):
    '''
    UNet implementation with transposed convolutions. The input size to a unet should be multiple of
    32x+220 where x is in N. This implementation is slightly modified from original paper and outputs
    same dimensional response maps as input.
    '''

    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        super(UNet, self).__init__(input_shape, learning_rate, no_classes, weight_file)

    def upconv2_2(self, input, concat_tensor, no_features):
        out_shape = [dim.value for dim in concat_tensor.get_shape()]
        up_conv = Deconvolution2D(no_features, 5, 5, out_shape, subsample=(2, 2))(input)
        # up_conv = Convolution2D(no_features, 2, 2)(UpSampling2D()(input))
        merged = merge([concat_tensor, up_conv], mode='concat', concat_axis=3)
        return merged

    def conv3_3(self, input, no_features):
        conv1 = Convolution2D(no_features, 3, 3, activation='relu')(input)
        conv2 = Convolution2D(no_features, 3, 3, activation='relu')(conv1)
        return conv2

    def build_model(self):
        input = Input(batch_shape=self.input_shape, name='input_1')
        conv1_1 = Convolution2D(64, 3, 3, activation='relu')(input)
        conv1_2 = Convolution2D(64, 3, 3, activation='relu')(conv1_1)
        conv1_out = MaxPooling2D()(conv1_2)

        conv2_1 = Convolution2D(128, 3, 3, activation='relu')(conv1_out)
        dropout2 = Dropout(0.5)(conv2_1)
        conv2_2 = Convolution2D(128, 3, 3, activation='relu')(dropout2)
        conv2_out = MaxPooling2D()(conv2_2)

        conv3_1 = Convolution2D(256, 3, 3, activation='relu')(conv2_out)
        dropout3 = Dropout(0.5)(conv3_1)
        conv3_2 = Convolution2D(256, 3, 3, activation='relu')(dropout3)
        conv3_out = MaxPooling2D()(conv3_2)

        conv4_1 = Convolution2D(512, 3, 3, activation='relu')(conv3_out)
        dropout4 = Dropout(0.5)(conv4_1)
        conv4_2 = Convolution2D(512, 3, 3, activation='relu')(dropout4)
        conv4_out = MaxPooling2D()(conv4_2)

        conv5_1 = Convolution2D(1024, 3, 3, activation='relu')(conv4_out)
        dropout5 = Dropout(0.5)(conv5_1)
        conv5_2 = Convolution2D(1024, 3, 3, activation='relu')(dropout5)
        conv5_out = MaxPooling2D()(conv5_2)

        up_conv1 = self.upconv2_2(conv5_out, conv4_out, 512)
        # conv6_out = self.conv3_3(up_conv1, 512)

        up_conv2 = self.upconv2_2(up_conv1, conv3_out, 256)
        # conv7_out = self.conv3_3(up_conv2, 256)

        up_conv3 = self.upconv2_2(up_conv2, conv2_out, 128)
        # conv8_out = self.conv3_3(up_conv3, 128)

        up_conv4 = self.upconv2_2(up_conv3, conv1_out, 64)
        # conv9_out = self.conv3_3(up_conv4, 64)

        out_shape = [dim.value for dim in input.get_shape()]
        out_shape = [self.batch_size] + out_shape[1:3] + [self.no_classes]
        output = Deconvolution2D(self.no_classes, 5, 5, out_shape, subsample=(2, 2), activation='sigmoid',
                                 name='class_out')(up_conv4)

        model = Model(input, output)
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy'}, metrics=['binary_accuracy'])
        model.summary()
        return model


if __name__ == '__main__':
    batch_size = 1
    detector = UNet([batch_size, 252, 252, 3], 1e-3, 1)

    dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations/', '0_bw.png', normalize=False)
    training_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'checkpoint_dir': '/data/cell_detection/test',
        'samples_per_epoc': 4,
        'nb_epocs': 500,
        'testing_ratio': 0.2,
        'validation_ratio': 0.1,
        'nb_validation_samples': 6

    }
    detector.train(**training_args)
