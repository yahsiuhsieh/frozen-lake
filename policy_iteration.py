import gym
import pybulletgym
import numpy as np

from utils import testPolicy, learnModel, plot


def policyEval(policy, value, trans_prob, reward, gamma, max_itr=20):
    """
    Policy evaluation

    : param policy: ndarray, given policy
    : param value: ndarray, given value function
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : return: updated value function
    """
    counter = 0
    num_state = policy.shape[0]

    while counter < max_itr:
        counter += 1
        for s in range(num_state):
            val = 0
            for s_new in range(num_state):
                val += trans_prob[s][policy[s]][s_new] * (
                    reward[s][policy[s]][s_new] + gamma * value[s_new]
                )
            value[s] = val
    return value


def policyImprove(policy, value, trans_prob, reward, gamma):
    """
    Policy improvement

    : param policy: ndarray, given policy
    : param value: ndarray, given value function
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : return:
        policy: updated policy
        policy_stable, bool, True if no change in policy
    """
    policy_stable = True
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[1]

    for s in range(num_state):
        old_action = policy[s]
        val = value[s]
        for a in range(num_action):
            tmp = 0
            for s_new in range(num_state):
                tmp += trans_prob[s][a][s_new] * (
                    reward[s][a][s_new] + gamma * value[s_new]
                )
            if tmp > val:
                policy[s] = a
                val = tmp
        if policy[s] != old_action:
            policy_stable = False
    return policy, policy_stable


def policyItr(trans_prob, reward, gamma=0.99, max_itr=30, stop_if_stable=False):
    """
    Policy iteration

    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : param stop_if_stable: bool, stop the training if reach stable state
    : return:
        policy: updated policy
        success_rate: list, success rate for each iteration
    """
    success_rate = []
    num_state = trans_prob.shape[0]

    # init policy and value function
    policy = np.zeros(num_state, dtype=int)
    value = np.zeros(num_state)

    counter = 0
    while counter < max_itr:
        counter += 1
        value = policyEval(policy, value, trans_prob, reward, gamma)
        policy, stable = policyImprove(policy, value, trans_prob, reward, gamma)

        # test the policy for each iteration
        success_rate.append(testPolicy(policy))

        if stable and stop_if_stable:
            print("policy is stable at {} iteration".format(counter))
    return policy, success_rate


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    env.reset()

    # toy policy
    policy = [(s + 1) % 4 for s in range(15)]
    print("Success rate of toy policy: {:.1f}%".format(testPolicy(policy) * 100))

    # get transitional probability and reward function
    trans_prob, reward = learnModel(env)

    # Policy Iteration
    PI_policy, PI_success_rate = policyItr(trans_prob, reward, max_itr=50)
    print("Final success rate of PI: {:.1f}%".format(PI_success_rate[-1] * 100))

    # plot
    plot(PI_success_rate, "Average success rate v.s. Episode (Policy Iteration)")
