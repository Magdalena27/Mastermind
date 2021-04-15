import pygame
import random
import copy
import numpy as np
from ValueIteration import ValueIteration
from FunctionApproximation import FA


class Mastermind:

    def __init__(self, code_to_decode):
        """
        Mastermind
        :param code_to_decode: 4-chars string of colors marks
        posiible colors: R (red), G (green), B (blue), Y (yellow)
        note: every color can be used only once
        """

        self.code_to_decode = code_to_decode
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 238, 0)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (156, 156, 156)

        self.display_mode_on = True

        self.cell_size = 60
        self.logo_image = pygame.transform.scale(pygame.image.load("graphics/mastermind_logo.png"), (int(5.5 * self.cell_size), int(self.cell_size)))
        self.line_image = pygame.transform.scale(pygame.image.load("graphics/line.JPG"), (int(5.5*self.cell_size), self.cell_size))
        pygame.init()
        self.screen = pygame.display.set_mode((int(5.5*self.cell_size), int(12.5*self.cell_size)))

        self.number_of_attempt = 0
        self.actual_attempt = None
        self.score = 0
        self.history_of_attempts = np.zeros(10).tolist()
        self.history_of_attempts_scores = []
        self.__draw_board()
        self.states = dict()
        self.possible_codes = []
        self.actions_to_try = []

    def reset(self):
        """ resets state of the environment"""
        self.history_of_attempts = np.zeros(10).tolist()
        self.score = 0
        self.number_of_attempt = 0
        return self.__get_state()

    def get_all_states(self):
        """ return a list of all possible states """
        if self.states != dict():
            return self.states
        else:
            number_of_states = 4*3*2*1
            state_name = []
            states = dict()

            for state in range(number_of_states):
                state_name.append(state)

            possible_codes = ["RYBG", "RYGB", "RGBY", "RBGY", "RBYG", "RGYB",
                              "GYBR", "GRBY", "GRYB", "GBYR", "GBRY", "GYRB",
                              "YGRB", "YGBR", "YBRG", "YBGR", "YRGB", "YRBG",
                              "BYGR", "BYRG", "BRYG", "BRGY", "BGYR", "BGRY"]

            for state in state_name:
                states[state] = possible_codes[state]

            self.states = states
            self.possible_codes = possible_codes
            self.actions_to_try = copy.deepcopy(possible_codes)

            return states

    def get_number_of_states(self):
        return len(self.get_all_states())

    def is_terminal(self, state):
        bp, wp = self.check_attempt(self.states[state])
        if bp == 4 or self.number_of_attempt == 10:
            return True
        else:
            return False

    def get_possible_actions(self, state):
        return self.actions_to_try

    def get_next_states(self, state, action):
        states = self.get_all_states()
        possible_states = []

        for s1 in states:
            if states[s1] in self.get_possible_actions(state):
                possible_states.append(s1)

        next_states = dict()
        for s in possible_states:

            # if self.is_terminal(s):
            #     next_states = dict()
            #     next_states[s] = 1
            #     return next_states
            # else:
            probability = 1. / len(possible_states)
            next_states[s] = probability

        return next_states

    def get_reward(self, state, action, next_state):
        bp, wp = self.check_attempt(self.states[next_state])
        each_step_points = -1
        if bp == 4:
            pin_combination_points = 1000
            game_over_points = 0
        else:
            pin_combination_points = bp*25 + wp * (-5)
            if self.number_of_attempt == 10:
                game_over_points = -1000
            else:
                game_over_points = 0

        # if action in self.actions_to_try:
        #     repeated_action_points = 0
        # else:
        #     repeated_action_points = -500

        reward = each_step_points + pin_combination_points + game_over_points #+ repeated_action_points

        return reward

    def step(self, action):
        '''
        :returns:
        state - current state of the game
        reward - reward received by taking action
        done - True if it is end of the game, False otherwise
        score - temporarily score of the game, later it will be displayed on the screen
        '''

        self.number_of_attempt += 1
        done = False

        # draw attempt and check
        self.history_of_attempts[self.number_of_attempt-1] = action
        self.actual_attempt = action
        self.__draw_board()

        # actualize score
        bp, wp = self.check_attempt(action)
        if bp == 4:
            self.score += 1000
            reward = 1000
            done = True
        else:
            attempt_score = bp*25 + wp*(-5) + self.number_of_attempt*(-1)
            self.score += attempt_score
            reward = attempt_score

            # check if there is another attempt
            if self.number_of_attempt == 10:
                self.score -= 1000
                reward -= 1000
                done = True

        # if action in self.actions_to_try:
        #     self.actions_to_try.remove(action)
        # else:
        #     reward -= 500
        #     self.score -= 500

        return self.__get_state(), reward, done, self.score

    def color_decoder(self, string_to_decode):
        color_tab = []
        for i in range(4):
            if string_to_decode[i] == "R":
                color_tab.append(self.red)
            elif string_to_decode[i] == "G":
                color_tab.append(self.green)
            elif string_to_decode[i] == "B":
                color_tab.append(self.blue)
            elif string_to_decode[i] == "Y":
                color_tab.append(self.yellow)
            else:
                raise ValueError("Not supported color!")
        return color_tab

    def check_attempt(self, attempt: str):

        black_pin_number = 0
        white_pin_number = 0
        for pin in range(len(attempt)):
            if attempt[pin] == self.code_to_decode[pin]:
                black_pin_number += 1
            elif attempt[pin] in self.code_to_decode:
                white_pin_number += 1
            else:
                pass
        return black_pin_number, white_pin_number

    def __draw_board(self):
        if self.display_mode_on:
            self.screen.fill((51, 51, 51))

            # logo
            self.screen.blit(self.logo_image, (0, 0))

            # code to decode
            rect = pygame.Rect(0, 75, 330, 60)
            pygame.draw.rect(self.screen, (47, 74, 54), rect)
            secret_code_color_tab = self.color_decoder(self.code_to_decode)
            for i in range(4):
                pygame.draw.circle(self.screen, secret_code_color_tab[i], (120 + i * 60, 105), 29)

            # pin holes
            score_bg = (145, 145, 145)
            for att_num in range(10):
                bg_rect = pygame.Rect(0, 150+att_num*60, 60, 60)
                pygame.draw.rect(self.screen, score_bg, bg_rect)
                for score_hole_row in range(2):
                    pygame.draw.circle(self.screen, self.black, (15+score_hole_row*30, 165+att_num*60), 6)
                    pygame.draw.circle(self.screen, self.black, (15+score_hole_row*30, 195+att_num*60), 6)
                for pin_hole in range(4):
                    pygame.draw.circle(self.screen, self.black, (120+pin_hole*60, 180+att_num*60), 6)

            # attempts
            for attempt in range(self.number_of_attempt):
                color_tab = self.color_decoder(self.history_of_attempts[attempt])
                for i in range(4):
                    pygame.draw.circle(self.screen, color_tab[i], (120 + i * 60, 720 - attempt * 60), 29)

            # attempts scores
            if self.actual_attempt is not None:
                black_pins_num, white_pins_num = self.check_attempt(self.actual_attempt)
                self.history_of_attempts_scores.append((black_pins_num, white_pins_num))

            for attempt in range(self.number_of_attempt):
                actual_pin_num = 0
                for bp in range(self.history_of_attempts_scores[attempt][0]):
                    actual_pin_num += 1
                    if actual_pin_num <= 2:
                        pygame.draw.circle(self.screen, self.black,
                                           (15 + (actual_pin_num-1) * 30, 705 - attempt * 60), 14)
                    elif actual_pin_num > 2:
                        pygame.draw.circle(self.screen, self.black,
                                           (15 + (actual_pin_num-3) * 30, 735 - attempt * 60), 14)
                for wp in range(self.history_of_attempts_scores[attempt][1]):
                    actual_pin_num += 1
                    if actual_pin_num <= 2:
                        pygame.draw.circle(self.screen, self.white,
                                           (15 + (actual_pin_num - 1) * 30, 705 - attempt * 60), 14)
                    elif actual_pin_num > 2:
                        pygame.draw.circle(self.screen, self.white,
                                           (15 + (actual_pin_num - 3) * 30, 735 - attempt * 60), 14)
            pygame.display.flip()

    def __get_state(self):
        states = self.get_all_states()
        for state in states:
            if states[state] == self.actual_attempt:
                return state
            else:
                return 0

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True


code_to_decode = "RBGY"

clock = pygame.time.Clock()

mastermind = Mastermind(code_to_decode)
mastermind.reset()

#Value Iteration
done = False
optimal_policy, optimal_value = ValueIteration(mastermind, 0.85, 0.00000000000000001)
state = mastermind.step("YGBR")[0]
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    '''
    try according to the policy
    '''
    actions = []
    probs = []

    for action, prob in optimal_policy[state].items():
        actions.append(action)
        probs.append(prob)

    # print(optimal_policy[state])
    state, reward, done, score = mastermind.step(np.random.choice(actions, 1, p=probs)[0])
    print(score)
    clock.tick(1)


def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)

        # train (update) agent for state
        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward

'''
Applying FA algorithm for Mastermind
'''
# agent = FA(alpha=0.001, epsilon=0.5, discount=0.99, env=mastermind)
#
# epocs = 5
# for i in range(epocs):
#     play_and_train(mastermind, agent)
#     print("Learned: ", (i / epocs) * 100, " %")
#     # print(agent.weights)
#     clock.tick(1)
# agent.turn_off_learning()
#
# cont = input("Continue?")
#
# if cont:
#
#     state = mastermind.reset()
#     done = False
#
#     clock.tick(5)
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 done = True
#
#         '''
#         play mastermind according to the policy from FA
#         '''
#         action = agent.get_action(state)
#
#         next_state, reward, done, score = mastermind.step(action)
#         agent.update(state, action, reward, next_state)
#         state = next_state
#         print(score)
#         # print(agent.weights)
#         # print(agent.get_qvalue(state, action))
#         clock.tick(5)