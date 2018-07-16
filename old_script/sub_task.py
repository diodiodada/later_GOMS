from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
import numpy as np
import tensorflow as tf
import gym
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '1'
set_session(tf.Session(config=config))


def get_data(filename):

    data = pickle.load(open(filename, 'rb'))

    # data normalization
    x = np.array([0, 5, 8, 11, 14, 17, 20, 27, 32, 35, 38, 41, 44, 47])
    y = x + 1
    z = x + 2

    data_x = data[:, x]
    data_y = data[:, y]
    data_z = data[:, z]

    data_x_mean = 1.3020565993173778
    data_x_std = 0.1208025863782076
    data_y_mean = 0.7487616786156264
    data_y_std = 0.12072576091100165
    data_z_mean = 0.4391706194317077
    data_z_std = 0.03664239363082237

    # data_x_mean = data_x.mean()
    # data_x_std = data_x.std()
    # data_y_mean = data_y.mean()
    # data_y_std = data_y.std()
    # data_z_mean = data_z.mean()
    # data_z_std = data_z.std()
    #
    # print(" data_x_mean:", data_x_mean, "\n",
    #       "data_x_std:", data_x_std, "\n",
    #       "data_y_mean:", data_y_mean, "\n",
    #       "data_y_std:", data_y_std, "\n",
    #       "data_z_mean:", data_z_mean, "\n",
    #       "data_z_std:", data_z_std, "\n")

    data_x = (data_x - data_x_mean) / data_x_std
    data_y = (data_y - data_y_mean) / data_y_std
    data_z = (data_z - data_z_mean) / data_z_std

    data[:, x] = data_x
    data[:, y] = data_y
    data[:, z] = data_z

    # count
    last_index = 0
    num_length = []
    num_index = []

    for i in range(data.shape[0]):
        if data[i, -1] == 1.0:
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

    data_reshape = np.zeros((num_length.shape[0], 100, 51))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]

    return data_reshape


def train_model():

    state = Input(shape=(100, 6))
    goal = Input(shape=(100, 3))

    concat_0 = Concatenate(axis=-1)([state, goal])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(100, input_shape=(50, 50), return_sequences=True, return_state=False, stateful=False)(concat_2)
    lstm_2 = LSTM(50, input_shape=(50, 100), return_sequences=True, return_state=False, stateful=False)(lstm_1)

    concat_3 = Dense(50, activation='relu')(lstm_2)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output = Dense(4)(concat_4)

    model = Model(inputs=[state, goal], outputs=output, name='behavior_cloning')

    return model


def test_model():

    state = Input(shape=(1, 6), batch_shape=(2, 1, 6))
    goal = Input(shape=(1, 3), batch_shape=(2, 1, 3))

    concat_0 = Concatenate(axis=-1)([state, goal])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(100, input_shape=(1, 50), return_sequences=True, return_state=False, stateful=True)(concat_2)
    lstm_2 = LSTM(50, input_shape=(1, 100), return_sequences=True, return_state=False, stateful=True)(lstm_1)

    concat_3 = Dense(50, activation='relu')(lstm_2)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output = Dense(4)(concat_4)

    model = Model(inputs=[state, goal], outputs=output, name='behavior_cloning')

    return model


def train(model):

    # get the data for training
    data = get_data("data/PP-1-paths-1000-[0].p")
    i = [0, 1, 2, 5, 6, 7]
    j = [11, 12, 13]
    state_feed = data[:, :, i]
    action_feed = data[:, :, 23:27]
    goal_feed = data[:, :, j]

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mean_squared_error',
                  # metrics=['mse'],
                  )

    tf_board = TensorBoard(log_dir='./logs-subtask',
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

    model_checkpoint = ModelCheckpoint('weights-subtask/subtask.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, goal_feed],
              action_feed,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=False,
              callbacks=[tf_board, model_checkpoint])


def test(model_for_25_nets):
    env = gym.make('FetchPickAndPlace-v0')

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss='mean_squared_error',
                              metrics=['mse'])

    model_for_25_nets.load_weights('weights-subtask/subtask.322-0.0109.hdf5', by_name=True)

    env.reset()
    env.step([0, 0, 0, 0])
    env.render()

    while True:

        two_state = np.zeros((2, 1, 6))
        two_goal = np.zeros((2, 1, 3))
        observation = env.reset()

        data = observation["my_new_observation"]

        x = np.array([0, 5, 8, 11, 14, 17, 20])
        y = x + 1
        z = x + 2

        data_x = data[x]
        data_y = data[y]
        data_z = data[z]

        data_x_mean = 1.3020565993173778
        data_x_std = 0.1208025863782076
        data_y_mean = 0.7487616786156264
        data_y_std = 0.12072576091100165
        data_z_mean = 0.4391706194317077
        data_z_std = 0.03664239363082237

        data_x = (data_x - data_x_mean) / data_x_std
        data_y = (data_y - data_y_mean) / data_y_std
        data_z = (data_z - data_z_mean) / data_z_std

        data[x] = data_x
        data[y] = data_y
        data[z] = data_z

        i = [0, 1, 2, 5, 6, 7]
        j = [11, 12, 13]

        two_state[0, 0, :] = data[i]
        two_goal[0, 0, :] = data[j]

        done = False

        model_for_25_nets.reset_states()

        while not done:
            env.render()
            action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
            action_raw = action_two[0, 0, :]
            action = np.zeros(4, )
            step_size = 0.01

            if round(action_raw[0]) == 0:
                action[0] = 0
            elif round(action_raw[0]) == 1:
                action[0] = -(step_size / 0.03)
            elif round(action_raw[0]) == 2:
                action[0] = (step_size / 0.03)

            if round(action_raw[1]) == 0:
                action[1] = 0
            elif round(action_raw[1]) == 1:
                action[1] = -(step_size / 0.03)
            elif round(action_raw[1]) == 2:
                action[1] = (step_size / 0.03)

            if round(action_raw[2]) == 0:
                action[2] = 0
            elif round(action_raw[2]) == 1:
                action[2] = -(step_size / 0.03)
            elif round(action_raw[2]) == 2:
                action[2] = (step_size / 0.03)

            if round(action_raw[3]) == 0:
                action[3] = -1
            elif round(action_raw[3]) == 1:
                action[3] = 1

            observation, reward, done, info = env.step(action)
            data = observation["my_new_observation"]

            x = np.array([0, 5, 8, 11, 14, 17, 20])
            y = x + 1
            z = x + 2

            data_x = data[x]
            data_y = data[y]
            data_z = data[z]

            data_x_mean = 1.3020565993173778
            data_x_std = 0.1208025863782076
            data_y_mean = 0.7487616786156264
            data_y_std = 0.12072576091100165
            data_z_mean = 0.4391706194317077
            data_z_std = 0.03664239363082237

            data_x = (data_x - data_x_mean) / data_x_std
            data_y = (data_y - data_y_mean) / data_y_std
            data_z = (data_z - data_z_mean) / data_z_std

            data[x] = data_x
            data[y] = data_y
            data[z] = data_z

            i = [0, 1, 2, 5, 6, 7]
            j = [11, 12, 13]

            two_state[0, 0, :] = data[i]
            two_goal[0, 0, :] = data[j]

        if done:
                print(True)


def check_usage_for_lstm(model_for_25_nets):
    # this file is used for check:
    # whether model loading weights correctly
    # whether model using the state from previous time

    data = pickle.load(open('FetchPickAndPlace-v0.p', 'rb'))

    data = data.reshape((5000, 50, 58))

    state_feed = data[:, :, 0:25]
    action_feed = data[:, :, 25:29]
    next_state_deed = data[:, :, 29:54]
    goal_feed = data[:, :, 54:57]
    done_feed = data[:, :, 57]

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss='mean_squared_error',
                              metrics=['mse'])

    model_for_25_nets.load_weights('FetchPickAndPlace.100-0.0072.hdf5', by_name=True)

    two_state = np.zeros((2, 1, 25))
    two_goal = np.zeros((2, 1, 3))

    two_state[0, 0, :] = state_feed[0, 0, :]
    two_state[1, 0, :] = state_feed[0, 0, :]

    two_goal[0, 0, :] = goal_feed[0, 0, :]
    two_goal[1, 0, :] = goal_feed[0, 0, :]

    action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
    print(action_two)

    print("\n")

    # model_for_25_nets.reset_states()

    action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
    print(action_two)


if __name__ == '__main__':

    # model = train_model()
    # train(model)


    model = test_model()
    test(model)
