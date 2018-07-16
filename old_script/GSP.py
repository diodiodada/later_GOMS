from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np


def train_model():

    state = Input(shape=(180, 23))
    goal = Input(shape=(180, 9))

    concat_0 = Concatenate(axis=-1)([state, goal])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)

    lstm_1 = LSTM(100, input_shape=(50, 50), return_sequences=True, return_state=False, stateful=False)(concat_2)

    concat_3 = Dense(50, activation='relu')(lstm_1)
    concat_4 = Dense(50, activation='relu')(concat_3)

    output_x = Dense(3, activation='softmax', name="x")(concat_4)
    output_y = Dense(3, activation='softmax', name="y")(concat_4)
    output_z = Dense(3, activation='softmax', name="z")(concat_4)
    output_hand = Dense(2, activation='softmax', name="hand")(concat_4)

    model = Model(inputs=[state, goal], outputs=[output_x, output_y, output_z, output_hand], name='behavior_cloning')

    return model


def test_model():

    state = Input(shape=(1, 23), batch_shape=(2, 1, 23))
    goal = Input(shape=(1, 9), batch_shape=(2, 1, 9))

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

    model = Model(inputs=[state, goal], outputs=[output_x, output_y, output_z, output_hand], name='behavior_cloning')

    return model


def train(model):

    # get the data for training
    data = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

    state_feed = data[:, :, 0:23]
    action_feed = data[:, :, 32:36]
    goal_feed = data[:, :, 23:32]

    action_x_feed = action_feed[:, :, 0]
    action_y_feed = action_feed[:, :, 1]
    action_z_feed = action_feed[:, :, 2]
    action_hand_feed = action_feed[:, :, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

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

    model_checkpoint = ModelCheckpoint('pick-place-push_category.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, goal_feed],
              [action_x_feed, action_y_feed, action_z_feed, action_hand_feed],
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
                              loss='categorical_crossentropy',
                              metrics=['accuracy'],
                              )

    model_for_25_nets.load_weights('ppp-category.967-0.6477.hdf5', by_name=True)

    while True:

        two_state = np.zeros((2, 1, 23))
        two_goal = np.zeros((2, 1, 9))
        observation = env.reset()
        two_state[0, 0, :] = observation["my_new_observation"][0:23]
        two_goal[0, 0, :] = observation["my_new_observation"][23:32]

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
            two_state[0, 0, :] = observation["my_new_observation"][0:23]
            two_goal[0, 0, :] = observation["my_new_observation"][23:32]

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
