import pickle
import gym
import numpy as np
import random
from PIL import Image
from itertools import permutations

from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


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


def decide_move(offset):
    global min
    global max
    plus = 2
    minus = 1
    fixed = 0
    action = [0, 0, 0]
    for i in range(3):
        if offset[i] > max:
            action[i] = plus
        elif offset[i] < min:
            action[i] = minus
        else:
            action[i] = fixed
    return action


def evalue_move(offset):
    global min
    global max
    if offset[0] < max and offset[1] < max and offset[2] < max and offset[0] > min and offset[1] > min and offset[
        2] > min:
        return True
    else:
        return False


def pick_and_place(position_claw, position_object, position_target):
    # suppose the claw already at standard height

    global stage
    global close_counter
    global open_counter
    global ground_height

    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0
    success = False

    action = [fixed, fixed, fixed, close]

    # stpe_1: reach_object_above (open hand)
    if stage == "reach_object_above":
        action_3 = position_object - position_claw + [0, 0, 0.1]

        if evalue_move(action_3):
            stage = "reach_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # step_2: reach_object (open hand)
    elif stage == "reach_object":

        action_3 = position_object - position_claw

        if evalue_move(action_3):
            stage = "grasp_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # step_3: grasp_object (close hand)
    elif stage == "grasp_object":
        if close_counter < 3:
            close_counter = close_counter + 1
        else:
            stage = "raise_object_up"
            close_counter = 0
        action[3] = close

    # step_4: raise_object_up (close hand)
    elif stage == "raise_object_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2]]

        if evalue_move(action_3):
            stage = "reach_target_above"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_5: reach_target_above (close hand)
    elif stage == "reach_target_above":

        action_3 = position_target - position_object + [0, 0, 0.1]

        if evalue_move(action_3):
            stage = "lower_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_6: lower_object (close hand)
    elif stage == "lower_object":

        action_3 = position_target - position_object + [0, 0, 0.04]

        if evalue_move(action_3):
            stage = "release_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_7: release_object (open hand)
    elif stage == "release_object":

        if open_counter < 3:
            open_counter = open_counter + 1
        else:
            stage = "raise_claw_up"
            open_counter = 0
        action[3] = hand_open

    # step_8: raise_claw_up (open hand)
    elif stage == "raise_claw_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2]]

        if evalue_move(action_3):
            stage = "inside_finish"
            success = True
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    return action, success


def subtask_decide(strategy_id,
                   object_0_position, object_1_position,
                   bow_0_position, bow_1_position,
                   goal_0_position, goal_1_position):
    if strategy_id == 0:
        return object_0_position, bow_0_position
    elif strategy_id == 1:
        return object_1_position, bow_1_position
    elif strategy_id == 2:
        return bow_0_position, goal_0_position
    elif strategy_id == 3:
        return bow_1_position, goal_1_position


def check(object_0_position, object_1_position,
          bow_0_position, bow_1_position,
          goal_0_position, goal_1_position):
    a = object_0_position - bow_0_position
    b = object_1_position - bow_1_position
    c = bow_0_position - goal_0_position
    d = bow_1_position - goal_1_position

    result = np.concatenate((a, b, c, d), axis=-1)

    for i in range(result.shape[0]):
        if result[i] < 0.1:
            pass
        else:
            return False
    return True


# Hyper parameters
step_size = 0.01

# global parameters
close_counter = 0
open_counter = 0
min = -step_size
max = step_size
ground_height = 0.0
stage = "reach_object_above"

env = gym.make('FetchPickAndPlace-v0')


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


def make_trajectory(strategy, desired_num, metric):
    global stage
    global ground_height

    trajectory_num = 0

    image_num_already_success = 0

    metric.compile(optimizer=Adam(lr=1e-4), loss='mse')
    metric.load_weights('weights-m/1-M.5966-5.16955442.hdf5')

    two_metric_state = np.zeros((2, 17))
    two_metric_goal = np.zeros((2, 6))

    rollout_times = 0

    while rollout_times < 10:
        rollout_times += 1
        # begins a new trajectory

        stage = "reach_object_above"
        stage_outside = "step_1"

        observation = env.reset()
        done = False

        gripper_position = observation["my_new_observation"][0:3]
        object_0_position = observation["my_new_observation"][5:8]
        object_1_position = observation["my_new_observation"][8:11]
        bow_0_position = observation["my_new_observation"][11:14]
        bow_1_position = observation["my_new_observation"][14:17]
        goal_0_position = observation["my_new_observation"][17:20]
        goal_1_position = observation["my_new_observation"][20:23]

        plus = 2
        minus = 1
        fixed = 0
        hand_open = 1
        close = 0

        ground_height = object_0_position[2]

        one_trajectory = []

        metric.reset_states()

        score_array = []

        while not done:

            # NOT saving image
            env.render()

            # ==================================================

            two_state_top = np.zeros((2, 1, 21))

            data = observation["my_new_observation"].copy()
            data = data_normalize(data)

            ii = [0, 1, 2,
                  5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22]
            two_state_top[0, 0, :] = data[ii]

            two_metric_state[0, :] = data[0:17]
            two_metric_goal[0, :] = data[17:23]

            score = metric.predict_on_batch([two_metric_state, two_metric_goal])
            D_1 = data[5:8] - data[17:20]
            D_2 = data[11:14] - data[17:20]
            D_3 = data[8:11] - data[20:23]
            D_4 = data[14:17] - data[20:23]
            D_1 = abs(D_1[0]) + abs(D_1[1]) + abs(D_1[2])
            D_2 = abs(D_2[0]) + abs(D_2[1]) + abs(D_2[2])
            D_3 = abs(D_3[0]) + abs(D_3[1]) + abs(D_3[2])
            D_4 = abs(D_4[0]) + abs(D_4[1]) + abs(D_4[2])
            D = D_1 + D_2 + D_3 + D_4

            score_array.append([score[0, 0], D])

            # ==================================================

            if stage_outside == "step_1":
                ob, tar = subtask_decide(strategy[0],
                                         object_0_position, object_1_position,
                                         bow_0_position, bow_1_position,
                                         goal_0_position, goal_1_position)
                action_category, success = pick_and_place(gripper_position, ob, tar)
                if success:
                    stage_outside = "step_2"
                    stage = "reach_object_above"
            elif stage_outside == "step_2":
                ob, tar = subtask_decide(strategy[1],
                                         object_0_position, object_1_position,
                                         bow_0_position, bow_1_position,
                                         goal_0_position, goal_1_position)
                action_category, success = pick_and_place(gripper_position, ob, tar)
                if success:
                    stage_outside = "step_3"
                    stage = "reach_object_above"
            elif stage_outside == "step_3":
                ob, tar = subtask_decide(strategy[2],
                                         object_0_position, object_1_position,
                                         bow_0_position, bow_1_position,
                                         goal_0_position, goal_1_position)
                action_category, success = pick_and_place(gripper_position, ob, tar)
                if success:
                    stage_outside = "step_4"
                    stage = "reach_object_above"
            elif stage_outside == "step_4":
                ob, tar = subtask_decide(strategy[3],
                                         object_0_position, object_1_position,
                                         bow_0_position, bow_1_position,
                                         goal_0_position, goal_1_position)
                action_category, success = pick_and_place(gripper_position, ob, tar)
                if success:
                    stage_outside = "outside_finish"
                    stage = "reach_object_above"

            action = np.zeros((4))

            # change from category to value
            for i in range(3):
                if action_category[i] == plus:
                    action[i] = (step_size / 0.03)
                elif action_category[i] == minus:
                    action[i] = -(step_size / 0.03)
                elif action_category[i] == fixed:
                    action[i] = 0.0

            if action_category[3] == hand_open:
                action[3] = 1.0
            elif action_category[3] == close:
                action[3] = -1.0

            observation, reward, done, info = env.step(action)

            gripper_position = observation["my_new_observation"][0:3]
            object_0_position = observation["my_new_observation"][5:8]
            object_1_position = observation["my_new_observation"][8:11]
            bow_0_position = observation["my_new_observation"][11:14]
            bow_1_position = observation["my_new_observation"][14:17]
            goal_0_position = observation["my_new_observation"][17:20]
            goal_1_position = observation["my_new_observation"][20:23]

            if stage_outside == "outside_finish":

                if check(object_0_position, object_1_position,
                         bow_0_position, bow_1_position,
                         goal_0_position, goal_1_position):
                    trajectory_num += 1
                    image_num_already_success += len(one_trajectory)

                break

        array = np.array(score_array)
        pickle.dump(array, open("expert-score-" + str(rollout_times) + ".p", "wb"))

        print(trajectory_num)

        if trajectory_num == desired_num:
            break


metric = metric_net()

make_trajectory([0, 1, 2, 3], 1000, metric)











