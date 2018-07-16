from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np


def generate_data():
    # generate training data

    num_trajectory = 5000

    data = pickle.load(open('FetchPickAndPlace-category-5000.p', 'rb'))
    data = data.reshape((5000, 50, 58))
    # fetch two trajectory
    data = data[0:num_trajectory, 0:30, :]
    data_ob = data[:, :, 0:6]
    data_goal = data[:, :, 54:57]
    data = np.concatenate((data_ob, data_goal), axis=-1)
    print(data.shape)

    train_x = []
    train_y = []

#                               go up   go up  go down
# let true value from [ -step*i, ..., 0, ..., 1, ..., 1-(29-j)*step ]
    for num_tra in range(num_trajectory):
        for i in range(29):
            for j in range(29, 30):
                step = 1.0/(j-i)
                start = -step * i
                middle = 1
                end = 1 - (29 - j) * step

                y_a = np.linspace(start, middle, num=j+1)
                y_b = np.linspace(middle, end, num=30-j)
                y_b = y_b[1:]
                y = np.concatenate((y_a, y_b), axis=-1)

                s_t = data[num_tra, i]
                s_g = data[num_tra, j]
                for k in range(30):
                    s_x = data[num_tra, k]
                    sample_x = [s_t, s_g, s_x]
                    sample_y = y[k]

                    train_x.append(sample_x)
                    train_y.append(sample_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(train_x.shape)
    print(train_y.shape)

    pickle.dump(train_x, open("measure_x.p", "wb"))
    pickle.dump(train_y, open("measure_y.p", "wb"))


def measure_network():

    s_t = Input(shape=(9,))
    s_g = Input(shape=(9,))
    s_x = Input(shape=(9,))

    concat_0 = Concatenate(axis=-1)([s_t, s_g, s_x])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)
    concat_3 = Dense(50, activation='relu')(concat_2)
    drop_1 = Dropout(0.5)(concat_3)
    concat_4 = Dense(50, activation='relu')(drop_1)
    concat_5 = Dense(50, activation='relu')(concat_4)
    concat_6 = Dense(50, activation='relu')(concat_5)
    output = Dense(1)(concat_6)

    model = Model(inputs=[s_t, s_g, s_x], outputs=output)

    return model


def train(model):

    # get the data for training
    train_x = pickle.load(open('measure_x.p', 'rb'))
    train_y = pickle.load(open('measure_y.p', 'rb'))

    print(train_x.shape)
    print(train_y.shape)

    s_t = train_x[:, 0, :]
    s_g = train_x[:, 1, :]
    s_x = train_x[:, 2, :]

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mean_squared_error',
                  # metrics=['mse'],
                  )

    # model.load_weights('FetchPickAndPlace.199-0.0034.hdf5', by_name=True)

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('measure_regression.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([s_t, s_g, s_x],
              train_y,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=False,
              callbacks=[tf_board, model_checkpoint])


generate_data()
model = measure_network()
train(model)


