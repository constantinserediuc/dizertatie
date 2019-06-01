from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
import os
class Discriminator(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = 1
        self.model = self.build_model()
        self.load_mnist()
        # self.initial_training()
        # self.model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '/initial_weights.h5')
    def load_mnist(self):
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 255
        self.x_train = np.expand_dims(X_train, axis=3)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape))  # (image_height, image_width, num_channels)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def initial_training(self):
        valid = to_categorical(np.ones((60000, 1)))

        results = self.model.fit(self.x_train, valid,
                            epochs=2, batch_size=1024)
        self.model.save_weights('initial_weights.h5')

    def train(self, binary_img):
        idx = np.random.randint(0, self.x_train.shape[0], self.batch_size)
        imgs = self.x_train[idx]
        imgs = np.vstack((imgs, binary_img))
        valid = to_categorical(np.ones((self.batch_size, 1))) #[0,1] este ok
        invalid = np.flip(to_categorical(np.ones((binary_img.shape[0], 1))),1)
        labels = np.vstack((valid,invalid))
        self.model.fit(imgs,labels, epochs=2, batch_size=self.batch_size, verbose=0)
