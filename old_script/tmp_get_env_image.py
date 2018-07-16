import pickle
import gym
import numpy as np
from PIL import Image

env = gym.make('Ant-v2')

observation = env.reset()
print(observation)

for i in range(10):
    a = env.render(mode='rgb_array')

    a = Image.fromarray(a)
    a.show()

    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    # print(observation['achieved_goal'])
    # next_state = observation["observation"]



