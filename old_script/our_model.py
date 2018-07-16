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


def remove_last_9(x):
    return x[:, :-9, :]


def remove_first_9(x):
    return x[:, 9:, :]


def get_data(filename, usage):

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

    if usage == "forward":
        return data

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

    data_reshape = np.zeros((num_length.shape[0], 400, 51))
    step_reshape = np.zeros((num_length.shape[0], 400, 1))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]
        step_reshape[i, 0:num_length[i], 0] = np.linspace(num_length[i]-1, 0, num_length[i])

    step = []
    for i in range(num_length.shape[0]):
        one_tra = np.linspace(num_length[i]-1, 0, num_length[i])
        step.extend(one_tra)

    step = np.array(step)

    if usage == "metrics":
        return data, step

    return data_reshape, step_reshape


# ======== define model ========
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


def forward_net():
    state = Input(shape=(17,))
    action = Input(shape=(11,))

    f_concat_0 = Concatenate(axis=-1)([state, action])

    f_concat_1 = Dense(64, activation='relu', name='f_concat_1')(f_concat_0)
    f_concat_2 = Dense(64, activation='relu', name='f_concat_2')(f_concat_1)
    f_concat_3 = Dense(64, activation='relu', name='f_concat_3')(f_concat_2)
    f_concat_4 = Dense(64, activation='relu', name='f_concat_4')(f_concat_3)
    f_output_state = Dense(17, activation=None, name="f_output_state")(f_concat_4)

    model = Model(inputs=[state, action], outputs=f_output_state, name='forward_net')

    return model


def our_model():
    state = Input(shape=(400, 17))
    goal = Input(shape=(400, 6))

    # =========== backward_model ===========
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_masking_0 = Masking(mask_value=0.0)(b_concat_0)

    b_concat_1 = Dense(64, activation='relu', name="b_concat_1")(b_masking_0)
    b_concat_2 = Dense(64, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(128, return_sequences=True, return_state=False, stateful=False, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(64, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(64, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="b_output_x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="b_output_y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="b_output_z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="b_output_hand")(b_concat_4)

    action_estimate = Concatenate(axis=-1)([b_output_x, b_output_y, b_output_z, b_output_hand])

    # =========== forward_model ===========
    f_concat_0 = Concatenate(axis=-1)([state, action_estimate])

    f_concat_1 = Dense(64, activation='relu', name='f_concat_1', trainable=False)(f_concat_0)
    f_concat_2 = Dense(64, activation='relu', name='f_concat_2', trainable=False)(f_concat_1)
    f_concat_3 = Dense(64, activation='relu', name='f_concat_3', trainable=False)(f_concat_2)
    f_concat_4 = Dense(64, activation='relu', name='f_concat_4', trainable=False)(f_concat_3)
    f_output_state = Dense(17, activation=None, name="f_output_state", trainable=False)(f_concat_4)

    # =========== metrics_model ===========
    m_concat_0 = Concatenate(axis=-1)([state, goal])
    m_concat_1 = Concatenate(axis=-1)([f_output_state, goal])

    m_masking_0 = Masking(mask_value=0.0)
    m_dense_1 = Dense(64, activation='relu', name='m_dense_1', trainable=False)
    m_dense_2 = Dense(64, activation='relu', name='m_dense_2', trainable=False)
    m_dense_3 = Dense(64, activation='relu', name='m_dense_3', trainable=False)
    m_output_step = Dense(1, activation=None, name="m_output_step", trainable=False)

    m_s0_mask = m_masking_0(m_concat_0)
    m_s0_1 = m_dense_1(m_s0_mask)
    m_s0_2 = m_dense_2(m_s0_1)
    m_s0_3 = m_dense_3(m_s0_2)
    m_s0_out = m_output_step(m_s0_3)

    m_s1_mask = m_masking_0(m_concat_1)
    m_s1_1 = m_dense_1(m_s1_mask)
    m_s1_2 = m_dense_2(m_s1_1)
    m_s1_3 = m_dense_3(m_s1_2)
    m_s1_out = m_output_step(m_s1_3)

    m_s0_removed = Lambda(remove_last_9)(m_s0_out)
    m_s1_removed = Lambda(remove_first_9)(m_s1_out)

    m_sub_out = Subtract()([m_s0_removed, m_s1_removed])

    model = Model(inputs=[state, goal],
                  outputs=[b_output_x,
                           b_output_y,
                           b_output_z,
                           b_output_hand,
                           f_output_state,
                           m_sub_out],
                  name='our_model')
    return model


def our_model_for_test():

    state = Input(shape=(1, 17), batch_shape=(2, 1, 17))
    goal = Input(shape=(1, 6), batch_shape=(2, 1, 6))

    # =========== backward_model ===========
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_concat_1 = Dense(64, activation='relu', name="b_concat_1")(b_concat_0)
    b_concat_2 = Dense(64, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(128, return_sequences=True, return_state=False, stateful=True, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(64, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(64, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="b_output_x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="b_output_y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="b_output_z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="b_output_hand")(b_concat_4)

    model = Model(inputs=[state, goal], outputs=[b_output_x, b_output_y, b_output_z, b_output_hand])

    return model


# ======== train model ========
def train_forward_net(model):
    data = get_data('data/PP-1-paths-1000-[0, 1, 2, 3].p', reshape=False)

    # get state feed
    state_feed = data[:, 0:17]

    # get next state feed
    next_state_feed = data[:, 27:44]

    # get action feed
    action_feed = data[:, 23:27]
    action_x_feed = action_feed[:, 0]
    action_y_feed = action_feed[:, 1]
    action_z_feed = action_feed[:, 2]
    action_hand_feed = action_feed[:, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

    action_feed = np.append(action_x_feed, action_y_feed, axis=-1)
    action_feed = np.append(action_feed, action_z_feed, axis=-1)
    action_feed = np.append(action_feed, action_hand_feed, axis=-1)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse',
                  )

    tf_board = TensorBoard(log_dir='./logs-f',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('weights-f/F.{epoch:d}-{val_loss:.8f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, action_feed],
              next_state_feed,
              batch_size=10000,
              # initial_epoch=201,
              epochs=500,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def train_metrics_net(model):
    data, step = get_data('data/PP-1-paths-1000-[0, 1, 2, 3].p', usage="metrics")

    state_feed = data[:, 0:17]
    goal_feed = data[:, 17:23]
    step_feed = step

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse',
                  )

    tf_board = TensorBoard(log_dir='./logs-m',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('weights-m/M.{epoch:d}-{val_loss:.8f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, goal_feed],
              step_feed,
              batch_size=10000,
              # initial_epoch=652,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def train_our_model(model):
    # data = reshape_data('Pick-Place-Push-category-4-paths-1000.p')
    data, step = get_data('data/PP-1-paths-1000-[0, 1, 2, 3].p', usage="")

    # ==== get state feed ====
    state_feed = data[:, :, 0:17]

    # ==== get goal feed ====
    goal_feed = data[:, :, 17:23]

    # ==== get action feed ====
    action_feed = data[:, :, 23:27]
    action_x_feed = action_feed[:, :, 0]
    action_y_feed = action_feed[:, :, 1]
    action_z_feed = action_feed[:, :, 2]
    action_hand_feed = action_feed[:, :, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

    # ==== get next state feed ====
    next_state_feed = data[:, :, 27:44]

    # ==== get score feed ====
    score_feed = np.full((data.shape[0], data.shape[1]-9, 1), 10)
    # score_feed = step

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mse',
                        'mse'],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc',
                           'f_output_state': 'mse',
                           'm_output_score': 'mse'},
                  loss_weights=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                  )

    model.load_weights('weights-i/2-I.999-0.28402096.hdf5', by_name=True)
    model.load_weights('weights-f/1-F.500-0.00023557.hdf5', by_name=True)
    model.load_weights('weights-m/1-M.5966-5.16955442.hdf5', by_name=True)

    tf_board = TensorBoard(log_dir='./logs-zs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('weights-zs/T.{epoch:d}-{val_loss:.8f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed,
               goal_feed],
              [action_x_feed,
               action_y_feed,
               action_z_feed,
               action_hand_feed,
               next_state_feed,
               score_feed],
              batch_size=400,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


# ======== test model ========
def test_our_model(model):
    step_size = 0.01

    env = gym.make('FetchPickAndPlace-v0')

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc'},
                  )

    # model.load_weights('weights-t/1-T.999-200.22514343.hdf5', by_name=True)
    model.load_weights('weights-i/2-I.999-0.28402096.hdf5', by_name=True)

    while True:

        two_state = np.zeros((2, 1, 17))
        two_goal = np.zeros((2, 1, 6))
        observation = env.reset()

        env.step([0, 0, 0, 0])
        env.render()
        observation = env.reset()

        # =====================================

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

        two_state[0, 0, :] = data[0:17]
        two_goal[0, 0, :] = data[17:23]

        # =====================================

        done = False

        model.reset_states()

        while not done:
            env.render()
            two_x, two_y, two_z, two_hand = model.predict_on_batch([two_state, two_goal])

            x = two_x[0, 0, :]
            y = two_y[0, 0, :]
            z = two_z[0, 0, :]
            hand = two_hand[0, 0, :]

            action = np.zeros(4, )

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

            two_state[0, 0, :] = data[0:17]
            two_goal[0, 0, :] = data[17:23]

            if done:
                print(True)


# ======== check model ========
def check_usage_for_lstm(model):
    # this file is used for check:
    # whether model loading weights correctly
    # whether model using the state from previous time

    data = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, :, i]
    goal_feed = data[:, :, 23:26]

    model.compile(optimizer=Adam(lr=1e-4),
                              loss=['categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy', ],

                              metrics={'b_output_x': 'acc',
                                       'b_output_y': 'acc',
                                       'b_output_z': 'acc',
                                       'b_output_hand': 'acc'},
                              )

    model.load_weights('our.944-4.472422.hdf5', by_name=True)

    two_state = np.zeros((2, 1, 11))
    two_goal = np.zeros((2, 1, 3))

    two_state[0, 0, :] = state_feed[0, 0, :]
    two_state[1, 0, :] = state_feed[0, 0, :]

    two_goal[0, 0, :] = goal_feed[0, 0, :]
    two_goal[1, 0, :] = goal_feed[0, 0, :]

    action_two = model.predict_on_batch([two_state, two_goal])
    print(action_two)

    print("\n")

    model.reset_states()

    action_two = model.predict_on_batch([two_state, two_goal])
    print(action_two)


# forward_model = forward_net()
# train_forward_net(forward_model)


# metrics_model = metric_net()
# train_metrics_net(metrics_model)


model = our_model()
train_our_model(model)


# import gym
# model = our_model_for_test()
# test_our_model(model)


# model = our_model_for_test()
# check_usage_for_lstm(model)

