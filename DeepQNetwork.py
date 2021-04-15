from collections import deque
import numpy as np
import random


class DQNAgent:
    def __init__(self, action_size, learning_rate, model):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = learning_rate
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        # Function adds information to the memory about last action and its results
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # get action in a given state (according to epsilon greedy algorithm)

        number_of_possible_actions = self.action_size

        # If there are no legal actions, return None
        if number_of_possible_actions == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        # get action in a given state (according to epsilon greedy algorithm)
        if number_of_possible_actions == 1:
            chosen_action = self.get_best_action(state)

        if number_of_possible_actions > 1:
            best = self.get_best_action(state)
            rest = list(np.arange(number_of_possible_actions))
            rest.remove(best)
            chosen_action_but_not_the_best_one = np.random.choice(rest)
            chosen_action = \
                np.random.choice([best, chosen_action_but_not_the_best_one], 1, p=[1 - epsilon, epsilon])[0]

        return chosen_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """
        number_of_possible_actions = self.action_size

        # get best possible action in a given state (remember to break ties randomly)

        if number_of_possible_actions == 0:
            return None

        #         actions_values = model.predict(state.reshape(1,64)).flatten()
        actions_values = self.model.predict(state).flatten()

        bests = []
        maximum_action_value = max(actions_values)
        for action in range(number_of_possible_actions):
            if actions_values[action] == maximum_action_value:
                bests.append(action)

        if len(bests) == 1:
            best_action = bests[0]
        else:
            best_action = np.random.choice(bests)

        return best_action

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory.
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for chosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        """
        # train network
        expected = []
        agent_history = []
        next_states = []

        agent_history_index = random.sample(range(0, len(self.memory)), batch_size)

        for index in agent_history_index:
            agent_history.append(self.memory[index][0])
            next_states.append(self.memory[index][3])

        agent_history = np.asarray(agent_history)
        next_states = np.asarray(next_states)

        # print(agent_history.shape, next_states.shape)
        last_i = 0
        predicted = self.model.predict(agent_history)
        expected = predicted
        prediction_for_next_states = self.model.predict(next_states)

        # print(prediction_for_next_states.shape)

        for index in agent_history_index:
            if self.memory[index][4]:
                expected[last_i][self.memory[index][1]] = self.memory[index][2]
            else:
                expected[last_i][self.memory[index][1]] = self.memory[index][2] + self.gamma * max(
                    prediction_for_next_states[last_i])
            last_i += 1
        agent_hist_arr = np.asarray(agent_history)
        expected_arr = np.asarray(expected)
        # print('x: ', agent_hist_arr.shape, 'y: ', expected_arr.shape)
        self.model.fit(agent_hist_arr, expected_arr, verbose=0, epochs=1)

    def update_epsilon_value(self):
        # Every each epoch epsilon value should be updated according to equation:
        # self.epsilon *= self.epsilon_decay, but the updated value shouldn't be lower then epsilon_min value
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min