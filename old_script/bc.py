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


def get_reduced_data(filename):

    data = pickle.load(open(filename, 'rb'))

    # count trajectory length
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

    reduced_tra = []
    repeat_times = []

    for i in range(num_length.shape[0]):
        begin = num_index[i] - num_length[i] + 1
        end = num_index[i]

        last_action = data[begin, 23:27]
        action_repeat = 1

        reduced_tra.append(data[begin])

        for j in range(begin+1, end + 1):
            if (last_action == data[j, 23:27]).all():
                action_repeat += 1
            else:
                repeat_times.append(action_repeat)
                reduced_tra.append(data[j])
                last_action = data[j, 23:27]
                action_repeat = 1
        repeat_times.append(action_repeat)

    reduced_tra = np.array(reduced_tra)
    repeat_times = np.array(repeat_times)

    repeat_times = np.reshape(repeat_times, (repeat_times.shape[0], 1))

    reduced_tra = np.concatenate((reduced_tra, repeat_times), axis=1)

    pickle.dump(reduced_tra, open("data/PP-1-paths-1000-[0, 1, 2, 3]-reduced.p", "wb"))


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

    return data

    # count
    last_index = 0
    num_length = []
    num_index = []

    for i in range(data.shape[0]):
        if data[i, -2] == 1.0:
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

    data_reshape = np.zeros((num_length.shape[0], 80, 52))
    step_reshape = np.zeros((num_length.shape[0], 80, 1))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]
        step_reshape[i, 0:num_length[i], 0] = np.linspace(num_length[i] - 1, 0, num_length[i])

    return data_reshape, step_reshape


def bc_net():
    state = Input(shape=(23,))

    b_dense_1 = Dense(64, activation='relu', name="b_dense_1")(state)
    b_dense_2 = Dense(64, activation='relu', name="b_dense_2")(b_dense_1)

    bn_0 = BatchNormalization()(b_dense_2)

    b_dense_3 = Dense(64, activation='relu', name="b_dense_3")(bn_0)
    b_dense_4 = Dense(64, activation='relu', name="b_dense_4")(b_dense_3)

    b_dense_x_1 = Dense(128, activation='relu', name="b_dense_x_1")(b_dense_4)
    b_dense_x_2 = Dense(128, activation='relu', name="b_dense_x_2")(b_dense_x_1)

    bn_1 = BatchNormalization()(b_dense_x_2)

    b_dense_x_3 = Dense(128, activation='relu', name="b_dense_x_3")(bn_1)
    b_output_x = Dense(3, activation='softmax', name="b_output_x")(b_dense_x_3)

    b_dense_y_1 = Dense(128, activation='relu', name="b_dense_y_1")(b_dense_4)
    b_dense_y_2 = Dense(128, activation='relu', name="b_dense_y_2")(b_dense_y_1)

    bn_2 = BatchNormalization()(b_dense_y_2)

    b_dense_y_3 = Dense(128, activation='relu', name="b_dense_y_3")(bn_2)
    b_output_y = Dense(3, activation='softmax', name="b_output_y")(b_dense_y_3)

    b_dense_z_1 = Dense(128, activation='relu', name="b_dense_z_1")(b_dense_4)
    b_dense_z_2 = Dense(128, activation='relu', name="b_dense_z_2")(b_dense_z_1)

    bn_3 = BatchNormalization()(b_dense_z_2)

    b_dense_z_3 = Dense(128, activation='relu', name="b_dense_z_3")(bn_3)
    b_output_z = Dense(3, activation='softmax', name="b_output_z")(b_dense_z_3)

    b_dense_hand_1 = Dense(128, activation='relu', name="b_dense_hand_1")(b_dense_4)
    b_dense_hand_2 = Dense(128, activation='relu', name="b_dense_hand_2")(b_dense_hand_1)

    bn_4 = BatchNormalization()(b_dense_hand_2)

    b_dense_hand_3 = Dense(128, activation='relu', name="b_dense_hand_3")(bn_4)
    b_output_hand = Dense(2, activation='softmax', name="b_output_hand")(b_dense_hand_3)

    b_dense_times_1 = Dense(128, activation='relu', name="b_dense_times_1")(b_dense_4)
    b_dense_times_2 = Dense(128, activation='relu', name="b_dense_times_2")(b_dense_times_1)

    bn_5 = BatchNormalization()(b_dense_times_2)

    b_dense_times_3 = Dense(64, activation='relu', name="b_dense_times_3")(bn_5)
    b_output_times = Dense(1, activation='relu', name="b_output_times")(b_dense_times_3)

    model = Model(inputs=state,
                  outputs=[b_output_x,
                           b_output_y,
                           b_output_z,
                           b_output_hand,
                           b_output_times],
                  name='our_model')
    return model


# ======== train model ========
def train_bc_net(model):
    data = get_data('data/PP-1-paths-1000-[0, 1, 2, 3]-reduced.p')

    # get state feed
    state_feed = data[:, 0:23]

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

    # get times feed
    times_feed = data[:, 51]

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mse'],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc',
                           'b_output_times': 'mse'},
                  loss_weights=[20.0, 20.0, 20.0, 20.0, 1.0],
                  )

    tf_board = TensorBoard(log_dir='./logs-bc',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('weights-bc/BC.{epoch:d}-{val_loss:.8f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit(state_feed,
              [action_x_feed,
               action_y_feed,
               action_z_feed,
               action_hand_feed,
               times_feed],
              batch_size=10000,
              # initial_epoch=201,
              epochs=5000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


# ======== test model ========
def test_bc(model):
    step_size = 0.01

    env = gym.make('FetchPickAndPlace-v0')

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mse'],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc',
                           'b_output_times': 'mse'},
                  )

    model.load_weights('weights-bc/1-BC.999-19.96895679.hdf5', by_name=True)

    while True:

        two_state = np.zeros((2, 23))
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

        two_state[0, :] = data[0:23]

        # =====================================

        done = False

        model.reset_states()

        while not done:
            env.render()
            two_x, two_y, two_z, two_hand, times = model.predict_on_batch(two_state)

            x = two_x[0, :]
            y = two_y[0, :]
            z = two_z[0, :]
            hand = two_hand[0, :]

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

            for i in range(int(times[0])):
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

            two_state[0, :] = data[0:23]

            if done:
                print(True)


def count_t(filename):

    data = pickle.load(open(filename, 'rb'))

    data = data[:, 23:27]
    data_last = data[0]
    num = 0
    for i in range(1, data.shape[0]):
        if (data[i] == data_last).all():
           pass
        else:
            num += 1
        data_last = data[i]

    print(num/1000)

    return


model = bc_net()
train_bc_net(model)

# import gym
# test_bc(model)

# get_reduced_data("data/PP-1-paths-1000-[0, 1, 2, 3].p")
# get_data("data/PP-1-paths-1000-[0, 1, 2, 3]-reduced.p")
