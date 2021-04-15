import numpy as np


def ValueIteration(env, gamma, theta):
    """
        This function calculate optimal policy for the specified env game using Value Iteration approach:
        :param env:
            get_all_states - return the dictionary of all states available in the environment
            get_possible_actions - return the list of possible actions for the given state
            get_next_states - return the list of possible next states (dictionary of states) with probability
                              for transition from state by taking action into next state
            get_reward - return the reward after taking action in state and landing on next_state
        :param gamma: discount factor
        :param theta: stop condition (algorithm should stop when minimal difference between previous evaluation
                      of policy and current is smaller than theta
        :return: optimal policy (dictionary) and value function for the policy (float)
    """
    V_old = dict()
    policy = dict()

    for s in env.get_all_states():
        actions = env.get_possible_actions(s)
        action_prob = 1 / len(actions)
        policy[s] = dict()
        for a in actions:
            policy[s][a] = action_prob

    for current_state in env.get_all_states():
        V_old[current_state] = 0
    i = 0
    while True:
        print(i)

        i += 1
        V = dict()
        for s in env.get_all_states():
            V[s] = V_old[s]

        for s in env.get_all_states():
            values_for_action = dict()
            values_for_action[s] = dict()

            for a in policy[s]:
                values_for_action[s][a] = 0
                value_for_all_next_actions = 0
                next_states = env.get_next_states(s, a)

                for s2 in next_states:
                    value_for_one_next_action = policy[s][a] * next_states[s2] * (
                            env.get_reward(s, a, s2) + gamma * V[s2])
                    value_for_all_next_actions += value_for_one_next_action

                values_for_action[s][a] = value_for_all_next_actions

            maximum_action_value = max(values_for_action[s].values())
            V[s] = maximum_action_value

        delta = np.abs(np.array(list(V.values())) - np.array(list(V_old.values())))
        max_delta = max(delta)
        print(max_delta)

        if max_delta < theta:
            break
        else:
            V_old = V

    ################################################################################
    # Policy improvement

    for s in env.get_all_states():
        actions_value = dict()
        actions_value[s] = dict()

        for a in env.get_possible_actions(s):
            actions_value[s][a] = 0
            actions_for_all_next_states_value = 0

            for s2 in env.get_next_states(s, a):
                action_for_one_next_state_value = env.get_next_states(s, a)[s2] * (
                        env.get_reward(s, a, s2) + gamma * V[s2])
                actions_for_all_next_states_value += action_for_one_next_state_value

            actions_value[s][a] = actions_for_all_next_states_value

        maximum_action_value = max(actions_value[s].values())
        many_max_val = False

        for action, value in actions_value[s].items():
            if len(actions_value[s]) == 1:
                if value == maximum_action_value:
                    best_action = action
                    policy[s][best_action] = 1
            else:
                values = []
                for action1 in actions_value[s]:
                    values.append(actions_value[s][action1])

                values = sorted(values, reverse=True)
                if values[0] == values[1]:
                    many_max_val = True

                if many_max_val:
                    policy[s][action] = 1. / len(actions_value[s])
                else:
                    policy[s][action] = (1 - 0.99) / (len(actions_value[s]) - 1)
                    if value == maximum_action_value:
                        best_action = action
                        policy[s][best_action] = 0.99

    return policy, V
