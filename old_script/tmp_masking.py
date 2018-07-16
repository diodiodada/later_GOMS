from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
# import gym
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

if __name__ == "__main__":
    a = np.array([[[1, 1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3, 3]],

                  [[1, 1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2, 2],
                   [5, 5, 5, 5, 5, 5]],
                  ])

    b = np.array([[[6],
                   [12],
                   [18]],

                  [[6],
                   [12],
                   [18]],
                  ])

    input = Input(shape=(3, 6))
    mask = Masking(mask_value=6)(input)

    # out = TimeDistributed(Dense(1, activation='linear'))(mask)
    # out = LSTM(1, return_sequences=True, return_state=False, stateful=False)(mask)

    out = Dense(1, activation='linear', trainable=False)(mask)

    model = Model(inputs=input, outputs=out)

    # print(model.summary())

    # model.set_weights([np.array([[1.], [1.], [1.], [1.], [1.], [1.]], dtype=np.float32),
    #                    np.array([0.], dtype=np.float32)])

    # print('Weights')
    # print(model.get_weights())
    # q = model.predict(a)
    # print(q)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mae',
                  )

    model.set_weights([np.array([[1.], [1.], [1.], [1.], [1.], [1.]], dtype=np.float32),
                       np.array([0.], dtype=np.float32)])

    model.fit(a,
              b,
              batch_size=2,
              # initial_epoch=201,
              epochs=5,
              verbose=1,
              validation_split=0,
              shuffle=True)

