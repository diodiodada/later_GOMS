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

    num_trajectory = 30
    length_trajectory = 20

    data = pickle.load(open('FetchPickAndPlace-category-5000.p', 'rb'))
    data = data.reshape((5000, 50, 58))
    # fetch two trajectory
    data = data[0:num_trajectory, 0:length_trajectory, :]
    data_ob = data[:, :, 0:6]
    data_goal = data[:, :, 54:57]
    data = np.concatenate((data_ob, data_goal), axis=-1)
    print(data.shape)

    train_x = []
    train_y = []

    for num_tra in range(num_trajectory):
        for i in range(length_trajectory):
            for j in range(length_trajectory):

                s_1 = data[num_tra, i]
                s_2 = data[num_tra, j]
                s_g = data[num_tra, length_trajectory-1]

                if np.array_equal(s_1, s_2):
                    sample_y = [1, 0, 0]
                elif i < j:
                    sample_y = [0, 1, 0]
                elif i > j:
                    sample_y = [0, 0, 1]

                sample_x = [s_1, s_2, s_g]

                train_x.append(sample_x)
                train_y.append(sample_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(train_x.shape)
    print(train_y.shape)

    pickle.dump(train_x, open("measure_x_CT.p", "wb"))
    pickle.dump(train_y, open("measure_y_CT.p", "wb"))


def generate_data_binary():
    # generate training data

    num_trajectory = 300
    length_trajectory = 20

    data = pickle.load(open('FetchPickAndPlace-category-5000.p', 'rb'))
    data = data.reshape((5000, 50, 58))
    # fetch two trajectory
    data = data[0:num_trajectory, 0:length_trajectory, :]
    data_ob = data[:, :, 0:6]
    data_goal = data[:, :, 54:57]
    data = np.concatenate((data_ob, data_goal), axis=-1)
    print(data.shape)

    train_x = []
    train_y = []

    for num_tra in range(num_trajectory):
        for i in range(length_trajectory):
            for j in range(length_trajectory):

                s_1 = data[num_tra, i]
                s_2 = data[num_tra, j]

                if np.array_equal(s_1, s_2):
                    pass
                else:
                    if i < j:
                        sample_y = [1, 0]
                    elif i > j:
                        sample_y = [0, 1]

                    sample_x = [s_1, s_2]

                    train_x.append(sample_x)
                    train_y.append(sample_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(train_x.shape)
    print(train_y.shape)

    pickle.dump(train_x, open("measure_x_CT.p", "wb"))
    pickle.dump(train_y, open("measure_y_CT.p", "wb"))


def measure_network():

    s_1 = Input(shape=(9,))
    s_2 = Input(shape=(9,))
    s_g = Input(shape=(9,))

    concat_0 = Concatenate(axis=-1)([s_1, s_2, s_g])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)
    concat_3 = Dense(50, activation='relu')(concat_2)
    output = Dense(3, activation='softmax')(concat_3)

    model = Model(inputs=[s_1, s_2, s_g], outputs=output)

    return model


def measure_network_without_g():

    s_1 = Input(shape=(9,))
    s_2 = Input(shape=(9,))

    concat_0 = Concatenate(axis=-1)([s_1, s_2])

    concat_1 = Dense(200, activation='relu')(concat_0)
    concat_2 = Dense(200, activation='relu')(concat_1)
    concat_3 = Dense(200, activation='relu')(concat_2)
    concat_4 = Dense(50, activation='relu')(concat_3)
    output = Dense(2, activation='softmax')(concat_4)

    model = Model(inputs=[s_1, s_2], outputs=output)

    return model


def train(model):

    # get the data for training
    train_x = pickle.load(open('measure_x_CT.p', 'rb'))
    train_y = pickle.load(open('measure_y_CT.p', 'rb'))

    print(train_x.shape)
    print(train_y.shape)

    s_1 = train_x[:, 0, :]
    s_2 = train_x[:, 1, :]
    # s_g = train_x[:, 2, :]

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    # model.load_weights('FetchPickAndPlace.199-0.0034.hdf5', by_name=True)

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_acc',
                               patience=2,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('measure_regression.{epoch:02d}-{val_acc:.4f}.hdf5',
                                       monitor='val_acc',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([s_1, s_2],
              # [s_1, s_2, sg],
              train_y,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


generate_data_binary()
model = measure_network_without_g()
train(model)


