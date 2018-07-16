from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np


def train_model_1():

    state = Input(shape=(50, 6))
    goal = Input(shape=(50, 3))

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

    # state = Input(shape=(1, 25))
    # goal = Input(shape=(1, 3))

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
    data = pickle.load(open('FetchPickAndPlace-50000.p', 'rb'))
    data = data.reshape((50000, 50, 58))
    state_feed = data[:, :, 0:6]
    action_feed = data[:, :, 25:29]
    goal_feed = data[:, :, 54:57]

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

    model_checkpoint = ModelCheckpoint('FetchPickAndPlace_feature_reduced.{epoch:02d}-{val_loss:.4f}.hdf5',
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

    model_for_25_nets.load_weights('FetchPickAndPlace_feature_reduced.24-0.0159.hdf5', by_name=True)

    while True:

        two_state = np.zeros((2, 1, 6))
        two_goal = np.zeros((2, 1, 3))
        observation = env.reset()
        two_state[0, 0, :] = observation["observation"][0:6]
        two_goal[0, 0, :] = observation["desired_goal"]

        done = False

        model_for_25_nets.reset_states()

        while not done:
        # while True:
            env.render()
            action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
            action = action_two[0, 0, :]

            observation, reward, done, info = env.step(action)
            two_state[0, 0, :] = observation["observation"][0:6]
            two_goal[0, 0, :] = observation["desired_goal"]

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

    # model = train_model_1()
    # train(model)

    model = test_model()
    test(model)
