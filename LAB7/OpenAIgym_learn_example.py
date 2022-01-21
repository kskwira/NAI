"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""
import time
import random
import pickle
import gym
import numpy as np

# selecting game environment
env = gym.make('MsPacman-v0')


alpha = 0.618
G = 0  # score
env_features = 210 * 160  # size of the map
lr_rate = 0.81
gamma = 0.96

# zeroing the Q-table
Q = np.zeros([env_features, env.action_space.n])

# learning best steps in X iteration
for episode in range(1, 61):
    done = False
    G, reward = 0, 0
    state = env.reset()

    while not done:
        env.render()
        # deciding if the next move should be random or based on the "past record"
        if random.random() < (0.8 / (episode * .1)):  # take less random steps as you learn more about the game
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = np.argmax(Q[state[:, :, 0]])

        # update stats and steps with selected move
        state2, reward, done, info = env.step(action)

        # adding formula score into Q-table
        Q[state[:, :, 0], action] += alpha * (reward + np.max(Q[state2[:, :, 0]]) - Q[state[:, :, 0], action])

        # update the score
        G += reward

        # update current state of the game
        state = state2

    print('Episode {} Total Reward: {}'.format(episode, G))

# save the learned outcomes in a file
with open("MsPacman_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)

env.close()


# RANDOM AGENT PLAY

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())  # take a random action
#     time.sleep(0.02)
#
# env.close()


# FROZEN LAKE

# env = gym.make('FrozenLake-v1')
#
# epsilon = 0.9
# total_episodes = 10
# max_steps = 100
#
# lr_rate = 0.81
# gamma = 0.96
#
# Q = np.zeros((env.observation_space.n, env.action_space.n))
#
#
# def choose_action(state):
#     action = 0
#     if np.random.uniform(0, 1) < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(Q[state, :])
#     return action
#
#
# def learn(state, state2, reward, action):
#     predict = Q[state, action]
#     target = reward + gamma * np.max(Q[state2, :])
#     Q[state, action] = Q[state, action] + lr_rate * (target - predict)
#
#
# # Start
# for episode in range(total_episodes):
#     state = env.reset()
#     t = 0
#
#     while t < max_steps:
#         env.render()
#         action = choose_action(state)
#         state2, reward, done, info = env.step(action)
#         learn(state, state2, reward, action)
#         state = state2
#         t += 1
#
#         if done:
#             break
#
#         time.sleep(0.1)
#
# print(Q)
#
# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(Q, f)
#
