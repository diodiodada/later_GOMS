from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '1'
set_session(tf.Session(config=config))


def data_normalize(data):
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

    return data


def get_control_signal(x, y, z, hand):
    action = np.zeros(4, )
    step_size = 0.01

    if x.argmax() == 0:
        action[0] = 0
    elif x.argmax() == 1:
        action[0] = -(step_size / 0.03)
    elif x.argmax() == 2:
        action[0] = (step_size / 0.03)

    if y.argmax() == 0:
        action[1] = 0
    elif y.argmax() == 1:
        action[1] = -(step_size / 0.03)
    elif y.argmax() == 2:
        action[1] = (step_size / 0.03)

    if z.argmax() == 0:
        action[2] = 0
    elif z.argmax() == 1:
        action[2] = -(step_size / 0.03)
    elif z.argmax() == 2:
        action[2] = (step_size / 0.03)

    if hand.argmax() == 0:
        action[3] = -1.0
    elif hand.argmax() == 1:
        action[3] = 1.0

    return action


def get_data(filename):

    data = pickle.load(open(filename, 'rb'))

    # data normalization
    x = np.array([0, 5, 8, 11, 14, 17, 20])
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

    data_reshape = np.zeros((num_length.shape[0], 4, 25))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]

    return data_reshape


def train_model():

    state = Input(shape=(4, 21))

    concat_1 = Dense(50, activation='relu')(state)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(50, return_sequences=True, return_state=False, stateful=False)(concat_2)

    concat_3 = Dense(50, activation='relu')(lstm_1)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output_choice = Dense(4, activation='softmax', name="x")(concat_4)

    model = Model(inputs=state, outputs=output_choice, name='behavior_cloning')

    return model


def test_model():

    state = Input(shape=(1, 21), batch_shape=(2, 1, 21))

    concat_1 = Dense(50, activation='relu')(state)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(50, return_sequences=True, return_state=False, stateful=True)(concat_2)

    concat_3 = Dense(50, activation='relu')(lstm_1)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output_choice = Dense(4, activation='softmax', name="x")(concat_4)

    model = Model(inputs=state, outputs=output_choice, name='behavior_cloning')

    return model


def test_model_sub():

    state = Input(shape=(1, 6), batch_shape=(2, 1, 6))
    goal = Input(shape=(1, 3), batch_shape=(2, 1, 3))

    concat_0 = Concatenate(axis=-1)([state, goal])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(100, input_shape=(50, 50), return_sequences=True, return_state=False, stateful=True)(concat_2)

    concat_3 = Dense(50, activation='relu')(lstm_1)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output_x = Dense(3, activation='softmax', name="x")(concat_4)
    output_y = Dense(3, activation='softmax', name="y")(concat_4)
    output_z = Dense(3, activation='softmax', name="z")(concat_4)
    output_hand = Dense(2, activation='softmax', name="hand")(concat_4)
    output_end = Dense(2, activation='softmax', name="output_end")(concat_4)

    model = Model(inputs=[state, goal], outputs=[output_x, output_y, output_z, output_hand, output_end], name='behavior_cloning')

    return model


def train(model):

    # get the data for training
    data = get_data("data/PP-1-paths-1000-[0, 1, 2, 3]-top4.p")
    i = [0,  1,  2,
         5,  6,  7,  8,  9,  10,
         11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22]
    state_feed = data[:, :, i]
    action_feed = data[:, :, 23]

    # for i in range(action_feed.shape[1]):
    #     print(action_feed[190, i])
    #
    # return 0
    #
    # action_feed = to_categorical(action_feed, 4)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    # model.load_weights('FetchPickAndPlace.199-0.0034.hdf5', by_name=True)

    tf_board = TensorBoard(log_dir='./logs-top4',
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

    model_checkpoint = ModelCheckpoint('weights-top4/top4.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit(state_feed,
              action_feed,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=False,
              callbacks=[tf_board, model_checkpoint])


def test(top_model, sub_model, metric_model):
    env = gym.make('FetchPickAndPlace-v0')

    top_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    top_model.load_weights('weights-top4/top4.247-0.0000.hdf5')

    sub_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    sub_model.load_weights('weights-sub-end-random/sub-end-random.442-0.0675.hdf5')

    metric_model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    metric_model.load_weights('weights-m/1-M.5966-5.16955442.hdf5')

    env.reset()

    env.step([0, 0, 0, 0])
    env.render()

    rollout_times = 0

    while rollout_times < 10:
        rollout_times += 1

        two_state_top = np.zeros((2, 1, 21))

        two_state_sub = np.zeros((2, 1, 6))
        two_goal_sub = np.zeros((2, 1, 3))

        two_metric_state = np.zeros((2, 17))
        two_metric_goal = np.zeros((2, 6))

        observation = env.reset()

        data = observation["my_new_observation"]
        data = data_normalize(data)

        ii = [0,   1,  2,
              5,   6,  7,  8,  9, 10,
              11, 12, 13, 14, 15, 16,
              17, 18, 19, 20, 21, 22]
        two_state_top[0, 0, :] = data[ii]

        print(rollout_times)
        done = False

        top_model.reset_states()
        sub_model.reset_states()

        last_index = 0
        score_array = []
        while not done:

            env.render()

            two_choice = top_model.predict_on_batch(two_state_top)

            choice = two_choice[0, 0, :]

            state_index = [0, 0, 0, 0, 0, 0]
            goal_index = [0, 0, 0]

            if choice.argmax() == 0:
                print("============== step 1 ==============")
                sub_model.reset_states()
                state_index = [0, 1, 2, 5, 6, 7]
                goal_index = [11, 12, 13]

            elif choice.argmax() == 1:
                print("============== step 2 ==============")
                sub_model.reset_states()
                state_index = [0, 1, 2, 8, 9, 10]
                goal_index = [14, 15, 16]

            elif choice.argmax() == 2:
                print("============== step 3 ==============")
                sub_model.reset_states()
                state_index = [0, 1, 2, 11, 12, 13]
                goal_index = [17, 18, 19]

            elif choice.argmax() == 3:
                if last_index == 3:
                    break
                else:
                    last_index = 3
                print("============== step 4 ==============")
                sub_model.reset_states()
                state_index = [0, 1, 2, 14, 15, 16]
                goal_index = [20, 21, 22]

            while not done:
                env.render()
                two_state_sub[0, 0, :] = data[state_index]
                two_goal_sub[0, 0, :] = data[goal_index]

                two_metric_state[0, :] = data[0:17]
                two_metric_goal[0, :] = data[17:23]

                # =============== action ===============

                score = metric_model.predict_on_batch([two_metric_state, two_metric_goal])
                D_1 = data[5:8] - data[17:20]
                D_2 = data[11:14] - data[17:20]
                D_3 = data[8:11] - data[20:23]
                D_4 = data[14:17] - data[20:23]
                D_1 = abs(D_1[0]) + abs(D_1[1]) + abs(D_1[2])
                D_2 = abs(D_2[0]) + abs(D_2[1]) + abs(D_2[2])
                D_3 = abs(D_3[0]) + abs(D_3[1]) + abs(D_3[2])
                D_4 = abs(D_4[0]) + abs(D_4[1]) + abs(D_4[2])
                D = D_1 + D_2 + D_3 + D_4

                two_x, two_y, two_z, two_hand, two_end = sub_model.predict_on_batch([two_state_sub, two_goal_sub])

                x = two_x[0, 0, :]
                y = two_y[0, 0, :]
                z = two_z[0, 0, :]
                hand = two_hand[0, 0, :]
                end = two_end[0, 0, :]

                if end.argmax() == 1:
                    ii = [0, 1, 2,
                          5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16,
                          17, 18, 19, 20, 21, 22]

                    two_state_top[0, 0, :] = data[ii]

                    break

                score_array.append([score[0, 0], D])

                action = get_control_signal(x, y, z, hand)

                observation, reward, done, info = env.step(action)
                data = observation["my_new_observation"]
                data = data_normalize(data)

        array = np.array(score_array)
        pickle.dump(array, open("our-score-" + str(rollout_times) + ".p", "wb"))


def metric_net():
    state = Input(shape=(17,))
    goal = Input(shape=(6,))

    m_concat_0 = Concatenate(axis=-1)([state, goal])

    m_dense_1 = Dense(64, activation='relu', name='m_dense_1')(m_concat_0)
    m_dense_2 = Dense(64, activation='relu', name='m_dense_2')(m_dense_1)
    m_dense_3 = Dense(64, activation='relu', name='m_dense_3')(m_dense_2)
    m_output_step = Dense(1,  activation=None,  name="m_output_step")(m_dense_3)

    model = Model(inputs=[state, goal], outputs=m_output_step, name='metric_net')

    return model


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


# model = train_model()
# train(model)

import gym
sub = test_model_sub()
top = test_model()
metric = metric_net()
test(top, sub, metric)
