import gym
import pybulletgym
import numpy as np
import matplotlib.pyplot as plt


def testPolicy(policy, trials=100):
    """
    Get the average rate of successful episodes over given number of trials
    : param policy: function, a deterministic policy function
    : param trials: int, number of trials
    : return: float, average success rate
    """
    env = gym.make("FrozenLake-v0")
    env.reset()
    success = 0

    for _ in range(trials):
        done = False
        state = env.reset()
        while not done:
            action = policy[state]
            state, _, done, _ = env.step(action)
            if state == 15:
                success += 1

    avg_success_rate = success / trials
    return avg_success_rate


def learnModel(env, samples=1e5):
    """
    Get the transition probabilities and reward function

    : param env: object, gym environment
    : param samples: int, random samples
    : return:
        trans_prob: ndarray, transition probabilities p(s'|a, s)
        reward: ndarray, reward function r(s, a, s')
    """
    # get size of state and action space
    num_state = env.observation_space.n
    num_action = env.action_space.n

    trans_prob = np.zeros((num_state, num_action, num_state))
    reward = np.zeros((num_state, num_action, num_state))
    counter_map = np.zeros((num_state, num_action, num_state))

    counter = 0
    while counter < samples:
        state = env.reset()
        done = False

        while not done:
            random_action = env.action_space.sample()
            new_state, r, done, _ = env.step(random_action)
            trans_prob[state][random_action][new_state] += 1
            reward[state][random_action][new_state] += r

            state = new_state
            counter += 1

    # normalization
    for i in range(trans_prob.shape[0]):
        for j in range(trans_prob.shape[1]):
            norm_coeff = np.sum(trans_prob[i, j, :])
            if norm_coeff:
                trans_prob[i, j, :] /= norm_coeff

    counter_map[counter_map == 0] = 1  # avoid invalid division
    reward /= counter_map

    return trans_prob, reward


def plot(success_rate, title):
    """
    Plots for success rate over every iteration

    :param success_rate: list, a list of success rate
    :param title: str, plot title
    """
    plt.figure()
    plt.plot(success_rate)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.savefig(title + ".png", dpi=150)
    plt.show()