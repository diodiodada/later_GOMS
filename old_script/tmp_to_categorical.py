from keras.utils import to_categorical
import numpy as np
import pickle

data = pickle.load(open('FetchPickAndPlace-category-5000.p', 'rb'))
data = data.reshape((5000, 50, 58))

state_feed = data[:, :, 0:25]
action_feed = data[:, :, 25:29]
goal_feed = data[:, :, 54:57]

action_x_feed = action_feed[:, :, 0]
action_y_feed = action_feed[:, :, 1]
action_z_feed = action_feed[:, :, 2]
action_hand_feed = action_feed[:, :, 3]

print(action_x_feed.shape)
print(action_y_feed.shape)
print(action_z_feed.shape)
print(action_hand_feed.shape)


action_x_feed = to_categorical(action_x_feed, 3)
action_y_feed = to_categorical(action_y_feed, 3)
action_z_feed = to_categorical(action_z_feed, 3)
action_hand_feed = to_categorical(action_hand_feed, 2)

print(action_x_feed.shape)
print(action_y_feed.shape)
print(action_z_feed.shape)
print(action_hand_feed.shape)