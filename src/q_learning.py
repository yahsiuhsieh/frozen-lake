import gym
import pybulletgym
import numpy as np

from utils import testPolicy, learnModel, plot


def epsilonGreedyExplore(env, state, Q_table, e, episodes):
    """
    epsilon-greedy exploration stratedy

    : param env: object, gym environment
    : param state: int, current state
    : param Q_table: ndarray, Q table
    : param e: int, current episode
    : param episodes: int, total number of episodes
    : return: action: int, chosen action {0,1,2,3}
    """
    prob = 1 - e / episodes
    if np.random.rand() < prob:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
    return action


def softmaxExplore(env, state, Q_table, tau=1):
    """
    Softmax exploration stratedy

    : param env: object, gym environment
    : param state: int, current state
    : param Q_table: ndarray, Q table
    : param tau: int, parameter for softmax
    : return: action: int, chosen action {0,1,2,3}
    """
    num_action = env.action_space.n
    action_prob = np.zeros(num_action)
    denominator = np.sum(np.exp(Q_table[state, :] / tau))

    for a in range(num_action):
        action_prob[a] = np.exp(Q_table[state, a] / tau) / denominator
    action = np.random.choice([0, 1, 2, 3], 1, p=action_prob)[0]
    return action


def Qlearning(
    env, alpha, gamma, episodes=5000, evaluate_policy=True, strategy="epsilon-greedy"
):
    """
    Q learning

    : param env: object, gym environment
    : param episodes: int, training episode, defaults to 5000
    : param evaluate_policy: bool, flag to disable recording success rate
    : param strategy: string, different exploration strategy, 'epsilon-greedy' or 'softmax'
    : return:
        policy: ndarray, a deterministic policy
        success_rate: list, success rate for each episode
    """
    # get size of state and action space
    num_state = env.observation_space.n
    num_action = env.action_space.n

    # init
    success_rate = []
    policy = np.zeros(num_state, dtype=int)
    Q_table = np.random.rand(num_state, num_action)

    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # choose action, 'epsilon-greedy' or 'softmax'
            if strategy == "epsilon-greedy":
                action = epsilonGreedyExplore(env, state, Q_table, i, episodes)
            else:
                action = softmaxExplore(env, state, Q_table)

            new_state, reward, done, _ = env.step(action)

            # update Q table
            Q_table[state][action] += alpha * (
                reward + gamma * max(Q_table[new_state, :]) - Q_table[state][action]
            )
            state = new_state

        # get deterministic policy from Q table
        for s in range(num_state):
            policy[s] = np.argmax(Q_table[s, :])

        # get success rate of current policy
        if evaluate_policy:
            if i % 100 == 0:
                success_rate.append(testPolicy(policy))
    return policy, success_rate


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    env.reset()

    # test different alpha with fixed gamma(0.99)
    alphas = [0.05, 0.1, 0.25, 0.5]
    for alpha in alphas:
        _, success_rate = Qlearning(env, alpha=alpha, gamma=0.99)
        print(
            "alpha = {}, gamma = {}: {:.1f}%".format(
                alpha, 0.99, success_rate[-1] * 100
            )
        )
        plot(
            success_rate,
            "Average success rate v.s. Episode (alpha={}, gamma=0.99)".format(alpha),
        )

    # test different gamma with fixed alpha(0.05)
    gammas = [0.9, 0.95, 0.99]
    for gamma in gammas:
        _, success_rate = Qlearning(env, alpha=0.05, gamma=gamma)
        print(
            "alpha = {}, gamma = {}: {:.1f}%".format(
                0.05, gamma, success_rate[-1] * 100
            )
        )
        plot(
            success_rate,
            "Average success rate v.s. Episode (alpha=0.05, gamma={})".format(gamma),
        )
