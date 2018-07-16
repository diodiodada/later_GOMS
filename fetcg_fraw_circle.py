import gym
import numpy
env = gym.make('FetchPickAndPlace-v0')
env.reset()

for _ in range(20):
    env.render()
    env.step([1, 0, 0, 0])


while(True):
    for i in range(20):
        env.render()
        observation, reward, done, info = env.step([1, 1, 0, 0])
        gripper_position = observation["my_new_observation"][0:3]
        # print(gripper_position)

        # if i == 1:
        #     a1 = gripper_position[0]
        #     b1 = gripper_position[1]
        # if i == 10:
        #     a2 = gripper_position[0]
        #     b2 = gripper_position[1]
        # if i == 18:
        #     a3 = gripper_position[0]
        #     b3 = gripper_position[1]
        #     y = -((a1-a3)*(a1*a1-a2*a2)-(a1-a2)*(a1*a1-a3*a3)+(a1-a3)*(b1*b1-b2*b2)-(a1-a2)*(b1*b1-b3*b3))/((a1-a3)*(2*b2-2*b1)-(a1-a2)*(2*b3-2*b1))
        #     x = (a1*a1-a3*a3+(2*b3-2*b1)*y+b1*b1-b3*b3)/(2*a1-2*a3)
        #     print(x, y)

        d = numpy.sqrt((gripper_position[0] - 0.675)*(gripper_position[0] - 0.675) + (gripper_position[1] - 0.756)*(gripper_position[1] - 0.756))
        print(d)

    for j in range(20):
        env.render()
        observation, reward, done, info = env.step([1, -1, 0, 0])
        gripper_position = observation["my_new_observation"][0:3]
        # print(gripper_position)

        d = numpy.sqrt((gripper_position[0] - 0.675) * (gripper_position[0] - 0.675) + (gripper_position[1] - 0.756) * (
                    gripper_position[1] - 0.756))
        print(d)
