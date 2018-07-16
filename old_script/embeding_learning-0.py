from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
# import gym
import numpy as np
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '1'
set_session(tf.Session(config=config))


def reshape_data(filename):
    data = pickle.load(open(filename, 'rb'))

    # data normalization
    x = np.array([0, 5, 8, 11, 14, 17, 20, 27, 32, 35, 38, 41, 44, 47])
    y = x + 1
    z = x + 2

    data_x = data[:, x]
    data_y = data[:, y]
    data_z = data[:, z]

    print("data_x.mean:", data_x.mean())
    print("data_x.std:", data_x.std())

    print("data_y.mean:", data_y.mean())
    print("data_y.std:", data_y.std())

    print("data_z.mean:", data_z.mean())
    print("data_z.std:", data_z.std())

    data_x = (data_x - data_x.mean()) / data_x.std()
    data_y = (data_y - data_y.mean()) / data_y.std()
    data_z = (data_z - data_z.mean()) / data_z.std()

    data[:, x] = data_x
    data[:, y] = data_y
    data[:, z] = data_z

    # count
    num_trajectory = 0
    last_index = 0
    num_length = []
    num_index = []

    for i in range(data.shape[0]):
        if data[i, -1] == 1.0:
            num_trajectory += 1
            length = i - last_index + 1
            num_length.append(length)
            num_index.append(i)
            last_index = i + 1

    num_length = np.array(num_length)
    num_index = np.array(num_index)
    print("mean:", num_length.mean(),
          "variance:", num_length.var(),
          "max:", num_length.max(),
          "min:", num_length.min(),
          "length:", num_length.shape[0])

    data_reshape = np.zeros((num_length.shape[0], 400, 51))

    state_0 = np.zeros((num_length.shape[0], 51))
    state_g = np.zeros((num_length.shape[0], 51))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]
        state_0[i, :] = data[num_index[i] - num_length[i] + 1, :]
        state_g[i, :] = data[num_index[i], :]

    return data_reshape, state_0, state_g


def trajectory_encoder():
    states = Input(shape=(100, 17))
    state_0 = Input(shape=(17,))
    state_g = Input(shape=(17,))

    # =========== encoder ===========

    e_dense_1 = Dense(50, activation='relu', name="e_dense_1")(states)
    e_dense_2 = Dense(50, activation='relu', name="e_dense_2")(e_dense_1)

    e_lstm_1 = LSTM(100, return_sequences=False, stateful=False, name="e_lstm_1")(e_dense_2)

    e_dense_3 = Dense(50, activation='relu', name="e_dense_3")(e_lstm_1)
    e_dense_4 = Dense(24, activation='softmax', name="e_dense_4")(e_dense_3)

    # =========== decoder ===========

    d_concat_0 = Concatenate(axis=-1)([e_dense_4, state_0, state_g])

    d_dense_1 = Dense(170, activation='relu', name="d_dense_1")(d_concat_0)
    d_dense_2 = Dense(340, activation='relu', name="d_dense_2")(d_dense_1)

    d_bn_1 = BatchNormalization(name='d_bn_1')(d_dense_2)

    d_dense_3 = Dense(680, activation='relu', name="d_dense_3")(d_bn_1)
    d_dense_4 = Dense(680, activation='relu', name="d_dense_4")(d_dense_3)

    d_bn_2 = BatchNormalization(name='d_bn_2')(d_dense_4)

    d_dense_5 = Dense(1020, activation='relu', name="d_dense_5")(d_bn_2)
    d_output = Dense(1700, activation='relu', name="d_output")(d_dense_5)

    model = Model(inputs=[states, state_0, state_g],
                  outputs=d_output,
                  name='encoder_model')

    # print(model.summary())

    return model


def train_trajectory_encoder(model):
    data, state_0, state_g = reshape_data('data/PP-24-paths-24000.p')

    # ==== get feed data ====
    states_feed = data[:, 0::4, 0:17]
    state_0_feed = state_0[:, 0:17]
    state_g_feed = state_g[:, 0:17]
    output_feed = states_feed.reshape((states_feed.shape[0], 1700))

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse',
                  )

    tf_board = TensorBoard(log_dir='./logs-0',
                           histogram_freq=30,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('weights-0/tra-encoder.{epoch:d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([states_feed,
               state_0_feed,
               state_g_feed],
              output_feed,
              batch_size=500,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


model = trajectory_encoder()
train_trajectory_encoder(model)



