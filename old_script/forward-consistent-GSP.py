from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
import pickle
import gym
import numpy as np


def train_model():

    action = Input(shape=(180, 11))  # x:3,y:3,z:3,gripper:2
    state = Input(shape=(180, 11))   # grip:3,grippers:2,object:3,gasket:3
    goal = Input(shape=(180, 3))     # goal:3

    # =============== backward model ===============
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(b_concat_0)
    b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=False, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="hand")(b_concat_4)

    action_estimate = Concatenate(axis=-1)([b_output_x, b_output_y, b_output_z, b_output_hand])

    # =============== shared dense layers ===============
    shared_dense_1 = Dense(50, activation='relu')
    shared_dense_2 = Dense(50, activation='relu')
    shared_dense_3 = Dense(50, activation='relu')
    shared_dense_4 = Dense(50, activation='relu')

    # =============== forward as regularization ===============

    f_as_r_concat_0 = Concatenate(axis=-1)([state, action])

    f_as_r_concat_1 = shared_dense_1(f_as_r_concat_0)
    f_as_r_concat_2 = shared_dense_2(f_as_r_concat_1)
    f_as_r_concat_3 = shared_dense_3(f_as_r_concat_2)
    f_as_r_concat_4 = shared_dense_4(f_as_r_concat_3)
    f_as_r_output_state = Dense(11, name="f_as_r_output_state")(f_as_r_concat_4)

    # =============== forward as consistent ===============

    f_as_c_concat_0 = Concatenate(axis=-1)([state, action_estimate])

    f_as_c_concat_1 = shared_dense_1(f_as_c_concat_0)
    f_as_c_concat_2 = shared_dense_2(f_as_c_concat_1)
    f_as_c_concat_3 = shared_dense_3(f_as_c_concat_2)
    f_as_c_concat_4 = shared_dense_4(f_as_c_concat_3)
    f_as_c_output_state = Dense(11, name="f_as_c_output_state")(f_as_c_concat_4)

    model = Model(inputs=[action, state, goal],
                  outputs=[b_output_x, b_output_y, b_output_z, b_output_hand, f_as_r_output_state, f_as_c_output_state])

    return model


def test_model():

    state = Input(shape=(1, 11), batch_shape=(2, 1, 11))
    goal = Input(shape=(1, 3), batch_shape=(2, 1, 3))

    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(b_concat_0)
    b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=True, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="hand")(b_concat_4)

    model = Model(inputs=[state, goal], outputs=[b_output_x, b_output_y, b_output_z, b_output_hand])

    return model


def train(model):

    # get the data for training
    data = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, :, i]

    j = i + 32
    next_state_feed = data[:, :, j]

    goal_feed = data[:, :, 23:26]

    action_feed = data[:, :, 32:36]
    action_x_feed = action_feed[:, :, 0]
    action_y_feed = action_feed[:, :, 1]
    action_z_feed = action_feed[:, :, 2]
    action_hand_feed = action_feed[:, :, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

    action_feed = np.append(action_x_feed, action_y_feed, axis=-1)
    action_feed = np.append(action_feed, action_z_feed, axis=-1)
    action_feed = np.append(action_feed, action_hand_feed, axis=-1)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mse',
                        'mse',
                        ],
                  metrics={'x': 'acc',
                           'y': 'acc',
                           'z': 'acc',
                           'hand': 'acc',
                           'f_as_r_output_state': 'mse',
                           'f_as_c_output_state': 'mse'},
                  )

    # model.compile(optimizer=Adam(lr=1e-4),
    #               loss={'x': 'categorical_crossentropy',
    #                     'y': 'categorical_crossentropy',
    #                     'z': 'categorical_crossentropy',
    #                     'hand': 'categorical_crossentropy',
    #                     'dense_9': 'mse',
    #                     'dense_9': 'mse'},
    #               metrics={'x': 'categorical_crossentropy',
    #                        'y': 'categorical_crossentropy',
    #                        'z': 'categorical_crossentropy',
    #                        'hand': 'categorical_crossentropy',
    #                        'dense_9': 'mse',
    #                        'dense_9': 'mse'},
    #               )

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

    model_checkpoint = ModelCheckpoint('FCFR-GSP.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([action_feed, state_feed, goal_feed],
              [action_x_feed, action_y_feed, action_z_feed, action_hand_feed, next_state_feed, next_state_feed],
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=False,
              callbacks=[tf_board, model_checkpoint])


def test(model_for_25_nets):
    step_size = 0.01

    env = gym.make('FetchPickAndPlace-v0')

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss=['categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    ],
                              metrics={'x': 'acc',
                                       'y': 'acc',
                                       'z': 'acc',
                                       'hand': 'acc'},
                              )

    model_for_25_nets.load_weights('FCFR-GSP.1000-0.7871.hdf5', by_name=True)

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    while True:

        two_state = np.zeros((2, 1, 11))
        two_goal = np.zeros((2, 1, 3))
        observation = env.reset()
        two_state[0, 0, :] = observation["my_new_observation"][i]
        two_goal[0, 0, :] = observation["my_new_observation"][23:26]

        done = False

        model_for_25_nets.reset_states()

        while not done:
            env.render()
            two_x, two_y, two_z, two_hand = model_for_25_nets.predict_on_batch([two_state, two_goal])

            x = two_x[0, 0, :]
            y = two_y[0, 0, :]
            z = two_z[0, 0, :]
            hand = two_hand[0, 0, :]

            action = np.zeros(4,)

            if x.argmax() == 0:
                action[0] = 0
            elif x.argmax() == 1:
                action[0] = -(step_size/0.03)
            elif x.argmax() == 2:
                action[0] = (step_size/0.03)

            if y.argmax() == 0:
                action[1] = 0
            elif y.argmax() == 1:
                action[1] = -(step_size/0.03)
            elif y.argmax() == 2:
                action[1] = (step_size/0.03)

            if z.argmax() == 0:
                action[2] = 0
            elif z.argmax() == 1:
                action[2] = -(step_size/0.03)
            elif z.argmax() == 2:
                action[2] = (step_size/0.03)

            if hand.argmax() == 0:
                action[3] = -1.0
            elif hand.argmax() == 1:
                action[3] = 1.0

            observation, reward, done, info = env.step(action)
            two_state[0, 0, :] = observation["my_new_observation"][i]
            two_goal[0, 0, :] = observation["my_new_observation"][23:26]

            if done:
                print(True)


def check_usage_for_lstm(model_for_25_nets):
    # this file is used for check:
    # whether model loading weights correctly
    # whether model using the state from previous time

    data = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, :, i]
    goal_feed = data[:, :, 23:26]

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss=['categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    ],
                              metrics={'x': 'acc',
                                       'y': 'acc',
                                       'z': 'acc',
                                       'hand': 'acc'},
                              )

    # model_for_25_nets.load_weights('FCFR-GSP.29-2.0846.hdf5', by_name=True)

    two_state = np.zeros((2, 1, 11))
    two_goal = np.zeros((2, 1, 3))

    two_state[0, 0, :] = state_feed[0, 0, :]
    two_state[1, 0, :] = state_feed[0, 0, :]

    two_goal[0, 0, :] = goal_feed[0, 0, :]
    two_goal[1, 0, :] = goal_feed[0, 0, :]

    action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
    print(action_two)

    print("\n")

    model_for_25_nets.reset_states()

    action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
    print(action_two)


if __name__ == '__main__':

    model = train_model()
    train(model)

    # model = test_model()
    # test(model)


    # check_usage_for_lstm(model)
