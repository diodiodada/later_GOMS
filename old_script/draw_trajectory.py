
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


observation = pickle.load(open("data/PP-24-paths-2400-0.p", 'rb'))
observation = observation[0:400, :]

gripper_position = observation[:, 0:3]

object_0_position = observation[:, 5:8]
object_1_position = observation[:, 8:11]

bow_0_position = observation[:, 11:14]
bow_1_position = observation[:, 14:17]

goal_0_position = observation[:, 17:20]
goal_1_position = observation[:, 20:23]

fig = plt.figure()


def plot_2d_pic():

    for i in range(observation.shape[0]):
        X = [gripper_position[i, 0],
             object_0_position[i, 0],
             object_1_position[i, 0],
             bow_0_position[i, 0],
             bow_1_position[i, 0],
             goal_0_position[i, 0],
             goal_1_position[i, 0]]

        Y = [gripper_position[i, 1],
             object_0_position[i, 1],
             object_1_position[i, 1],
             bow_0_position[i, 1],
             bow_1_position[i, 1],
             goal_0_position[i, 1],
             goal_1_position[i, 1]]

        Z = [gripper_position[i, 2],
             object_0_position[i, 2],
             object_1_position[i, 2],
             bow_0_position[i, 2],
             bow_1_position[i, 2],
             goal_0_position[i, 2],
             goal_1_position[i, 2]]

        ax = fig.add_subplot(111)

        ax.scatter(X, Y)

        ax.set_xlim(1, 1.5)
        ax.set_ylim(0.5, 1)

        plt.savefig("data/" + str(i) + ".jpg")

        fig.clf()


def plot_3d_pic():

    for i in range(observation.shape[0]):
        X = [gripper_position[i, 0],
             object_0_position[i, 0],
             object_1_position[i, 0],
             bow_0_position[i, 0],
             bow_1_position[i, 0],
             goal_0_position[i, 0],
             goal_1_position[i, 0]]

        Y = [gripper_position[i, 1],
             object_0_position[i, 1],
             object_1_position[i, 1],
             bow_0_position[i, 1],
             bow_1_position[i, 1],
             goal_0_position[i, 1],
             goal_1_position[i, 1]]

        Z = [gripper_position[i, 2],
             object_0_position[i, 2],
             object_1_position[i, 2],
             bow_0_position[i, 2],
             bow_1_position[i, 2],
             goal_0_position[i, 2],
             goal_1_position[i, 2]]

        ax = Axes3D(fig)

        ax.scatter(X, Y, Z)

        ax.set_xlim(1, 1.5)
        ax.set_ylim(0.5, 1)
        ax.set_zlim(0.4, 0.6)

        plt.savefig("data/" + str(i) + ".jpg")

        # plt.show()
        fig.clf()


plot_3d_pic()

