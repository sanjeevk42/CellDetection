from keras.engine import Input
from keras.engine import Model
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Reshape
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.optimizers import Adam

from detection.dataset.image_dataset import ImageDataset
from detection.detectors.model_optimization import start_training


def cnn_conv3(include_top=False, weights='random', input_shape=(64, 64, 3), out_size=5):
    '''
    Network with three convolutional layers.
    '''
    input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(input)
    x = Convolution2D(16, 5, 5, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool1')(x)

    x = Convolution2D(32, 5, 5, name='conv2', border_mode='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool2')(x)
    x = Dropout(0.7)(x)

    x = Convolution2D(64, 5, 5, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool3')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', name='fcn64')(x)
    x = Dropout(0.7)(x)

    class_out = Dense(1 * out_size, activation='sigmoid', name='class_out')(x)

    bb_out = Dense(5 * out_size, activation='relu')(x)
    bb_out = Reshape((out_size, 5), name='bb_out')(bb_out)

    model = Model(input, output=[class_out, bb_out])

    optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss={'class_out': 'binary_crossentropy', 'bb_out': 'mean_squared_error'})

    return model


if __name__ == '__main__':
    dataset_dir = '/data/lrz/hm-cell-tracking/sequences_A549/annotations'
    checkpoint_dir = '/data/training/cnn_conv3'
    batch_size = 20
    no_of_objects = 5
    input_shape = (64, 64, 3)
    samples_per_epoc = 80000
    nb_epocs = 1000

    model = cnn_conv3(input_shape=input_shape, out_size=no_of_objects)

    image_dataset = ImageDataset(dataset_dir)
    dataset_generator = image_dataset.patch_dataset_generator(batch_size, patch_size=input_shape[:2],
                                                              no_of_objects=no_of_objects, dataset='training')
    validation_generator = image_dataset.patch_dataset_generator(batch_size, patch_size=input_shape[:2],
                                                                 no_of_objects=no_of_objects, dataset='validation')
    nb_val_samples = 100
    start_training(dataset_dir, checkpoint_dir, model, dataset_generator, samples_per_epoc,
                   nb_epocs, validation_generator=validation_generator, nb_val_samples=nb_val_samples)
