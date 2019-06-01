from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Add
from keras.layers import TimeDistributed, LSTM
from keras.models import Model
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        image_input = Input(shape=(28, 28, 1))
        conv = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(image_input)

        action_input = Input(shape=(4,))
        dense = Dense(32, activation='relu')(action_input)  # cele 2 puncte din ultima actiune

        added = Add()([conv, dense])
        conv1 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same')(added)
        down_sampled1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        dropout1 = Dropout(0.25)(down_sampled1)
        conv2 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same')(dropout1)
        down_sampled2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # dimens (16,16,32)
        # after_conv = Add()([conv2,added])
        flatten = TimeDistributed(Flatten())(down_sampled2)

        lstm = LSTM(1, return_sequences=True)(flatten)
        after_lstm = TimeDistributed(Dense(1))(lstm)

        model = Model(inputs=[image_input, action_input], outputs=after_lstm)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        # print(model.summary())

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()
