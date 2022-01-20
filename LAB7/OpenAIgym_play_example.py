"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""
import os
import time
import pickle
import gym
import numpy as np

env = gym.make('MsPacman-v0')

with open("MsPacman_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)

for episode in range(1, 11):
    done = False
    G, reward = 0, 0
    state = env.reset()

    while not done:
        env.render()
        action = np.argmax(Q[state[:, :, 0]])
        state2, reward, done, info = env.step(action)
        G += reward
        state = state2

    print('Episode {} Total Reward: {}'.format(episode, G))

env.close()


# FROZEN LAKE
# env = gym.make('FrozenLake-v1')
#
# with open("frozenLake_qTable.pkl", 'rb') as f:
#     Q = pickle.load(f)
#
#
# def choose_action(state):
#     action = np.argmax(Q[state, :])
#     return action
#
#
# # start
# for episode in range(1):
#
#     state = env.reset()
#     print("*** Episode: ", episode)
#     t = 0
#     while t < 100:
#         env.render()
#         action = choose_action(state)
#         state2, reward, done, info = env.step(action)
#         state = state2
#
#         if done:
#             break
#
#         time.sleep(0.5)
#         os.system('clear')
#
# print(Q)
