import pickle
import gym
import numpy as np


def policy(observation):
    global stage
    global close_counter
    stage_change = "not-change"

    plus = 2
    minus = 1
    fixed = 0

    hand_open = 1
    close = 0

    action = [fixed, fixed, fixed, close]

    if stage == "reach_object":
        # move towards the object
        # if distance > 0.03 of distance < -0.03, using 1/-1
        # else using distance exactly
        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.07

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[1] > min and action_3[2] > min:
            # print("reach the target !!")
            stage = stage_set[1]
            stage_change = "from-reach-to-go-down"
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = plus
                elif action_3[i] < min:
                    action_3[i] = minus
                else:
                    action_3[i] = fixed
            action[0:3] = action_3
        action[3] = hand_open

    elif stage == "go_down":

        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.0

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[
            1] > min and action_3[2] > min:
            # print("go down already !!")
            stage = stage_set[2]
            stage_change = "from-go-down-to-close"
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = plus
                elif action_3[i] < min:
                    action_3[i] = minus
                else:
                    action_3[i] = fixed
            action[0:3] = action_3
        action[3] = hand_open

    elif stage == "close":
        # close the claw !!
        if close_counter < 3:
            close_counter = close_counter + 1
        else:
            # print("close the claw !!")
            stage = stage_set[3]
            close_counter = 0
            stage_change = "from-close-to-reach"
        action[3] = close

    elif stage == "reach":

        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        action_3 = desired_goal - achieved_goal

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[
            1] > min and action_3[2] > min:
            # print("reach already !!")
            stage = stage_set[4]
            stage_change = "already-to-goal"
            pass
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = plus
                elif action_3[i] < min:
                    action_3[i] = minus
                else:
                    action_3[i] = fixed
            action[0:3] = action_3
        action[3] = close

    elif stage == "hold":

        action[3] = close

    return action, stage_change


env = gym.make('FetchPickAndPlace-v0')

stage_set = ["reach_object", "go_down", "close", "reach", "hold"]
close_counter = 0

from_reach_to_go_down = []
from_go_down_to_close = []
from_close_to_reach = []
already_to_goal = []

for trajectory_num in range(10):
    stage = stage_set[0]

    observation = env.reset()
    done = False
    # print(observation)

    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0

    step = 1
    while not done:
        # env.render()

        action_category, stage_change = policy(observation)

        if stage_change == "from-reach-to-go-down":
            from_reach_to_go_down.append(step)
        elif stage_change == "from-go-down-to-close":
            from_go_down_to_close.append(step)
        elif stage_change == "from-close-to-reach":
            from_close_to_reach.append(step)
        elif stage_change == "already-to-goal":
            already_to_goal.append(step)



        action = np.zeros((4))
        for i in range(3):
            if action_category[i] == plus:
                action[i] = 0.5
            elif action_category[i] == minus:
                action[i] = -0.5
            elif action_category[i] == fixed:
                action[i] = 0.0

        if action_category[3] == hand_open:
            action[3] = 1.0
        elif action_category[3] == close:
            action[3] = -1.0

        observation, reward, done, info = env.step(action)

        step = step + 1

    print(trajectory_num)

print("==========================")

from_reach_to_go_down = np.array(from_reach_to_go_down)
from_go_down_to_close = np.array(from_go_down_to_close)
from_close_to_reach = np.array(from_close_to_reach)
already_to_goal = np.array(already_to_goal)

print(len(from_reach_to_go_down))
print(len(from_go_down_to_close))
print(len(from_close_to_reach))
print(len(already_to_goal))

print(from_reach_to_go_down.mean(), end=" ")
print(from_reach_to_go_down.std(), end=" ")
print(from_reach_to_go_down.min(), end=" ")
print(from_reach_to_go_down.max())

print(from_go_down_to_close.mean(), end=" ")
print(from_go_down_to_close.std(), end=" ")
print(from_go_down_to_close.min(), end=" ")
print(from_go_down_to_close.max())

print(from_close_to_reach.mean(), end=" ")
print(from_close_to_reach.std(), end=" ")
print(from_close_to_reach.min(), end=" ")
print(from_close_to_reach.max())

print(already_to_goal.mean(), end=" ")
print(already_to_goal.std(), end=" ")
print(already_to_goal.min(), end=" ")
print(already_to_goal.max())


