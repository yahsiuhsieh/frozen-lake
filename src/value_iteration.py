import gym
import pybulletgym
import numpy as np

from utils import testPolicy, learnModel, plot


def valueItr(trans_prob, reward, gamma=0.99, max_itr=100):
    """
    Value iteration

    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : return:
        policy: updated policy
        success_rate: list, success rate for each iteration
    """
    success_rate = []
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[1]

    # init policy and value function
    policy = np.zeros(num_state, dtype=int)
    value = np.zeros(num_state)

    counter = 0
    while counter < max_itr:
        counter += 1

        # value update
        for s in range(num_state):
            val = 0
            for a in range(num_action):
                tmp = 0
                for s_new in range(num_state):
                    tmp += trans_prob[s][a][s_new] * (
                        reward[s][a][s_new] + gamma * value[s_new]
                    )
                val = max(val, tmp)
            value[s] = val

        # policy recovery
        for s in range(num_state):
            val = 0
            for a in range(num_action):
                tmp = 0
                for s_new in range(num_state):
                    tmp += trans_prob[s][a][s_new] * (
                        reward[s][a][s_new] + gamma * value[s_new]
                    )
                if tmp > val:
                    policy[s] = a
                    val = tmp

        # test the policy for each iteration
        success_rate.append(testPolicy(policy))
    return policy, success_rate


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    env.reset()

    # toy policy
    policy = [(s + 1) % 4 for s in range(15)]
    print("Success rate of toy policy: {:.1f}%".format(testPolicy(policy) * 100))

    # get transitional probability and reward function
    trans_prob, reward = learnModel(env)

    # Value Iteration
    VI_policy, VI_success_rate = valueItr(trans_prob, reward, max_itr=50)
    print("Final success rate of VI: {:.1f}%".format(VI_success_rate[-1] * 100))

    # plot
    plot(VI_success_rate, "Average success rate v.s. Episode (Value Iteration)")