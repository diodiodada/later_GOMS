import pickle
import gym
import numpy as np
import random
from PIL import Image
from itertools import permutations


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


def pick_and_place(position_claw, position_object, position_target, finger_space):
    global stage
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
        if finger_space >= 0.07:
            pass
        else:
            stage = "raise_object_up"
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

        action_3 = position_target - position_object + [0, 0, 0.01]

        if evalue_move(action_3):
            stage = "release_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_7: release_object (open hand)
    elif stage == "release_object":

        if finger_space < 0.09:
            pass
        else:
            stage = "raise_claw_up"
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


def pick_and_place_only_if(position_claw, position_object, position_target, finger_space):
    global stage
    global last_stage
    global ground_height

    fixed = 0
    hand_open = 1
    close = 0
    success = False

    action = [fixed, fixed, fixed, close]

    e1 = abs(position_object - position_target)
    e2 = abs(position_claw - position_object)
    e3 = abs(position_claw[2] - ground_height - 0.1)            # 目标高度1
    e4 = abs(position_target[2] + 0.01 - position_object[2])    # 目标高度2
    # 物体不在目标点
    if not(e1[0] < 0.01 and e1[1] < 0.01 and e1[2] < 0.01):
        # 物体没被抓住
        if not(finger_space < 0.07 and e2[0] < 0.012 and e2[1] < 0.01 and e2[2] < 0.01):
            # 物体不在爪子xy范围
            if not(e2[0] < 0.012 and e2[1] < 0.01):
                # 爪子不在目标高度1
                if not(e3 < 0.01):
                    stage = "raise_claw_up"
                # 爪子在目标高度1
                else:
                    stage = "reach_object_above"
            # 物体在爪子xy范围
            else:
                # 物体不在爪子z范围
                if not(e2[2] < 0.01):
                    stage = "reach_object"
                # 物体在爪子z范围
                else:
                    stage = "grasp_object"
        # 物体被抓住
        else:
            # 物体不在目标xy范围
            if not (e1[0] < 0.01 and e1[1] < 0.01):
                # 物体不在目标高度1
                if not (e3 < 0.01):
                    stage = "raise_object_up"
                # 物体到达目标高度1
                else:
                    stage = "reach_target_above"
            # 物体在目标xy范围
            else:
                # 物体未到达目标高度2
                if not(e4 < 0.01):
                    stage = "lower_object"
                # 物体到达目标高度2
                else:
                    stage = "release_object"
    # 物体在目标点
    else:
        # 爪子不在目标高度1
        if not (e3 < 0.01):
            stage = "raise_claw_up"
        # 爪子在目标高度1
        else:
            stage = "inside_finish"
            success = True

    # print(stage)
    # print(e2)
    # if last_stage == "reach_target_above" and stage == "reach_object_above":
    #     exit(0)
    #
    # last_stage = stage

    if stage == "reach_object_above":
        action_3 = position_object - position_claw + [0, 0, 0.1]

        action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "reach_object":

        action_3 = position_object - position_claw

        action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "grasp_object":

        action[3] = close

    elif stage == "raise_object_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2]]

        action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "reach_target_above":

        action_3 = position_target - position_object + [0, 0, 0.1]
        action_3[2] = 0

        action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "lower_object":

        action_3 = position_target - position_object + [0, 0, 0.01]

        action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "release_object":

        action[3] = hand_open

    elif stage == "raise_claw_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2]]

        action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "inside_finish":
        action[3] = hand_open

    return action, success


def subtask_decide(strategy_id,
                   object_0_position, object_1_position,
                   bow_0_position, bow_1_position,
                   goal_0_position, goal_1_position):
    if strategy_id == 0:
        return object_0_position, bow_0_position + [0, 0, 0.03]
    elif strategy_id == 1:
        return object_1_position, bow_1_position + [0, 0, 0.03]
    elif strategy_id == 2:
        return bow_0_position, goal_0_position
    elif strategy_id == 3:
        return bow_1_position, goal_1_position


def subtask_decide_only_if(strategy_id,
                           object_0_position, object_1_position,
                           bow_0_position, bow_1_position,
                           goal_0_position, goal_1_position):

    # when object is on gasket, is taller than 0.0346
    e_0_1 = abs(object_0_position - bow_0_position)
    e_1_1 = abs(bow_0_position - goal_0_position)
    e_0_2 = abs(object_1_position - bow_1_position)
    e_2_2 = abs(bow_1_position - goal_1_position)

    f_0_1 = 0
    f_1_1 = 0
    f_0_2 = 0
    f_2_2 = 0

    if e_0_1[0] < 0.025 and e_0_1[1] < 0.02 and e_0_1[2] > 0.03 and e_0_1[2] < 0.04:
        f_0_1 = 1
    if e_1_1[0] < 0.01 and e_1_1[1] < 0.01 and e_1_1[2] < 0.01:
        f_1_1 = 1
    if e_0_2[0] < 0.025 and e_0_2[1] < 0.02 and e_0_2[2] > 0.03 and e_0_2[2] < 0.04:
        f_0_2 = 1
    if e_2_2[0] < 0.01 and e_2_2[1] < 0.01 and e_2_2[2] < 0.01:
        f_2_2 = 1

    # print(e_0_1)
    # print(e_1_1)
    # print(e_0_2)
    # print(e_2_2)
    #
    # print([f_0_1, f_1_1, f_0_2, f_2_2])

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [0, 0, 0, 0]:                # move first object
        return object_0_position, bow_0_position + [0, 0, 0.03], False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [1, 0, 0, 0]:                # move second object
        return object_1_position, bow_1_position + [0, 0, 0.03], False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [1, 0, 1, 0]:                # move first gasket
        return bow_0_position, goal_0_position, False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [1, 1, 1, 0]:                # move second gasket
        return bow_1_position, goal_1_position, False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [1, 1, 1, 1]:                # task finished
        return bow_1_position, goal_1_position, True

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [0, 1, 0, 1]:                # move first object
        return object_0_position, bow_0_position + [0, 0, 0.03], False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [0, 0, 1, 0]:                # move first object
        return object_0_position, bow_0_position + [0, 0, 0.03], False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [1, 1, 0, 0]:                # move second object
        return object_1_position, bow_1_position + [0, 0, 0.03], False

    if [f_0_1, f_1_1, f_0_2, f_2_2] == [0, 1, 1, 0]:                # move second object
        return object_0_position, bow_0_position + [0, 0, 0.03], False

    print([f_0_1, f_1_1, f_0_2, f_2_2])


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
min = -step_size
max = step_size
ground_height = 0.0
stage = "reach_object_above"
last_stage = ""

env = gym.make('FetchPickAndPlace-v0')


def make_trajectory(strategy, desired_num, render):
    global stage
    global ground_height

    data = []
    trajectory_num = 0

    image_num_already_success = 0

    while True:
        # begins a new trajectory

        stage = "reach_object_above"
        stage_outside = "step_1"

        observation = env.reset()
        done = False

        gripper_position = observation["my_new_observation"][0:3]
        finger_space = observation["my_new_observation"][3] + observation["my_new_observation"][4]
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
        final_success = False

        image_num = image_num_already_success

        # strategy = np.random.permutation(4)

        # to avoid all black image
        # env.render(mode='rgb_array')

        while True:

            # print(stage)
            # print("object_0_position:", object_0_position)
            # print("object_1_position:", object_1_position)
            # print("bow_0_position:", bow_0_position)
            # print("bow_1_position:", bow_1_position)

            # desktop height: 0.42, floor height:0.02
            if not(object_0_position[2] > 0.4 and object_1_position[2] > 0.4 and bow_0_position[2] > 0.4 and bow_1_position[2] > 0.4):
                print("object or gasket fall on the ground !")
                break

            d_0 = np.sqrt(
                (object_0_position[0] - 0.675) * (object_0_position[0] - 0.675) + (object_0_position[1] - 0.756) * (
                        object_0_position[1] - 0.756))
            d_1 = np.sqrt(
                (object_1_position[0] - 0.675) * (object_1_position[0] - 0.675) + (object_1_position[1] - 0.756) * (
                        object_1_position[1] - 0.756))
            d_2 = np.sqrt(
                (bow_0_position[0] - 0.675) * (bow_0_position[0] - 0.675) + (bow_0_position[1] - 0.756) * (
                        bow_0_position[1] - 0.756))
            d_3 = np.sqrt(
                (bow_1_position[0] - 0.675) * (bow_1_position[0] - 0.675) + (bow_1_position[1] - 0.756) * (
                        bow_1_position[1] - 0.756))

            if not(d_0 < 0.87 and d_1 < 0.87 and d_2 < 0.87 and d_3 < 0.87):
                print("object or gasket out of range !")
                break

            # saving image
            # image = env.render(mode='rgb_array')
            # image = Image.fromarray(image)
            # w, h = image.size
            # image = image.resize((w//4, h//4),Image.ANTIALIAS)
            # image.save('images/'+ str(image_num) +'.jpg', 'jpeg')
            # image_num += 1

            # NOT saving image
            if render:
                env.render()

            ob, tar, finish = subtask_decide_only_if(strategy,
                                                     object_0_position, object_1_position,
                                                     bow_0_position, bow_1_position,
                                                     goal_0_position, goal_1_position)
            action_category, success = pick_and_place_only_if(gripper_position, ob, tar, finger_space)
            if finish:
                stage_outside = "outside_finish"
                stage = "reach_object_above"
                final_success = True

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

            previous_observation = observation

            action_random = np.random.randint(0, 100)
            if action_random < 2:
                for _ in range(10):
                    env.step(env.action_space.sample())
                    if render:
                        env.render()
                observation, reward, done, info = env.step(env.action_space.sample())
            else:
                observation, reward, done, info = env.step(action)

            gripper_position = observation["my_new_observation"][0:3]
            finger_space = observation["my_new_observation"][3] + observation["my_new_observation"][4]
            object_0_position = observation["my_new_observation"][5:8]
            object_1_position = observation["my_new_observation"][8:11]
            bow_0_position = observation["my_new_observation"][11:14]
            bow_1_position = observation["my_new_observation"][14:17]
            goal_0_position = observation["my_new_observation"][17:20]
            goal_1_position = observation["my_new_observation"][20:23]

            one_step = []
            one_step.extend(previous_observation["my_new_observation"])
            one_step.extend(action_category)
            one_step.extend(observation["my_new_observation"])
            one_step.extend([float(final_success)])
            one_trajectory.append(one_step)

            if stage_outside == "outside_finish":

                trajectory_num += 1
                image_num_already_success += len(one_trajectory)
                print("task finish !")
                break

        print(trajectory_num)

        if trajectory_num == desired_num:
            break

    print("total trajectory num is : ", end="")
    print(image_num_already_success)

    data = np.array(data)
    pickle.dump(data, open("PP-1-paths-" + str(desired_num) + "-" + str(strategy) + "-random-init.p", "wb"))


# a = [0, 1, 2, 3]
# for perm in permutations(a):
#     print("strategy:", list(perm))
#     make_trajectory(list(perm), 20)


make_trajectory([0, 1, 2, 3], 1000, True)



