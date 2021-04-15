from collections import defaultdict
import numpy as np
import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class FA:
    def __init__(self, alpha, epsilon, discount, env):
        """
         FA Agent
         Instance variables
         - self.epsilon (exploration prob)
         - self.alpha (learning rate)
         - self.discount (discount rate aka gamma)
        """
        self.env = env
        self.get_legal_actions = env.get_possible_actions
        self.get_all_states = env.get_all_states
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.weights = np.array([0.1, 0.1])


    def get_maximum_value(self, state):
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        # get maximum possible value for a given state
        values = []
        for action in possible_actions:
             values.append(self.get_qvalue(state, action))

        max_value = max(values)
        return max_value

    def get_qvalue(self, state, action):
        weights = self.weights
        qvalue = weights[0] * self.how_many_black_pins(state, action) + weights[1] * self.no_white_pins(state, action)

        return qvalue

    def how_many_black_pins(self, state, action):
        bp, wp = self.env.check_attempt(action)
        return bp/4.0

    def no_white_pins(self, state, action):
        bp, wp = self.env.check_attempt(action)
        if wp == 0:
            return 1.0
        else:
            return 0.0

    def update(self, state, action, reward, next_state):
        """
         Weights update:
         delta = (r + gamma * maxa'Q(s',a')) - Q(s, a)
         wi = wi + alpha * delta * fi(s,a)
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        functions_list = [self.how_many_black_pins(state, action), self.no_white_pins(state, action)]

        # update weight for the given state and action
        delta = (reward + gamma * self.get_maximum_value(next_state)) - self.get_qvalue(state, action)
        for weight in range(len(self.weights)):
                    change = learning_rate * delta * functions_list[weight]
                    self.weights[weight] = self.weights[weight] + change

    def get_best_action(self, state):
        """
         Compute the best action to take in a state.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # get best possible action in a given state
        bests = []
        maximum_action_value = self.get_maximum_value(state)
        for action in possible_actions:
            if self.get_qvalue(state, action) == maximum_action_value:
                bests.append(action)

        if len(bests) == 1:
            best_action = bests[0]
        else:

            best_action = np.random.choice(bests)

        return best_action

    def get_action(self, state):
        """
         Compute the action to take in the current state, including exploration.
         With probability self.epsilon, we should take a random action.
             otherwise - the best policy action (self.get_best_action).
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        # get action in a given state (according to epsilon greedy algorithm)
        if len(possible_actions) == 1:
            chosen_action = possible_actions[0]

        if len(possible_actions) > 1:
            best = self.get_best_action(state)
            rest = possible_actions.copy()
            rest.remove(best)
            chosen_action_but_not_the_best_one = np.random.choice(rest)
            chosen_action = \
                np.random.choice([best, chosen_action_but_not_the_best_one], 1, p=[1 - epsilon, epsilon])[0]

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
