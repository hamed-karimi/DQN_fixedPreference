import numpy as np
import torch
from torch.nn import ReLU, Sigmoid
import random
from copy import deepcopy
from itertools import product


class Agent:
    def __init__(self, h, w, n, lambda_preference, prob_init_needs_equal, predefined_location,
                 preassigned_preferences, lambda_satisfaction,
                 rho_function='ReLU', epsilon_function='Linear'):  # n: number of needs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = h
        self.width = w
        self.location = self.initial_location(predefined_location)
        self.num_preference = n
        # self.initial_range_of_need = [-12, 12]
        # self.range_of_need = [-12, 12]
        # self.prob_init_needs_equal = prob_init_needs_equal
        self.preference = self.set_preference(preassigned_preferences)
        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.lambda_preference = lambda_preference  # How much the preference increases after each action
        self.lambda_satisfaction = lambda_satisfaction
        self.lambda_cost = 1
        self.no_reward_threshold = -5
        self.relu = ReLU()
        total_preference_functions = {'ReLU': self.relu, 'PolyReLU': self.poly_relu}
        self.rho_function = total_preference_functions[rho_function]
        # self.total_need = self.get_total_need()
        possible_h_w = [list(range(h)), list(range(w))]
        self.epsilon_function = epsilon_function
        self.all_locations = torch.from_numpy(np.array([element for element in product(*possible_h_w)]))

    def poly_relu(self, x, p=2):
        return self.relu(x) ** p

    def set_preference(self, preassigned_preferences=None):
        if any(preassigned_preferences):
            preference = torch.tensor(preassigned_preferences)
        else:
            preference = torch.zeros((1, self.num_preference))
            p = random.randint(0, 1)
            preference[0, p] = 1
            # preference = (self.initial_range_of_need[1] - self.initial_range_of_need[0]) * preference + self.initial_range_of_need[0]
        return preference

    def initial_location(self, predefined_location): # predefined_location is a list
        if len(predefined_location[0]) > 0:
            return torch.tensor(predefined_location)
        return torch.from_numpy(np.asarray((np.random.randint(self.height), np.random.randint(self.width)))).unsqueeze(0)

    # def update_preference_after_step(self, time_past):
    #     for i in range(self.num_preference):
    #         self.preference[0, i] += (self.lambda_preference * time_past)

    # def reward_function(self, preference):
        # x = preference.clone()
        # pos = self.relu(x)
        # # neg = (torch.pow(1.1, x)-1) * 7
        # neg = x
        # neg[x > 0] = 0
        # neg[x < self.no_reward_threshold] = self.no_reward_threshold #(pow(1.1, self.no_reward_threshold) - 1) * 7
        # return pos + neg
        # positive_part = torch.minimum(reward, self.relu(self.preference))
        # negative_part = sig_lin(self.preference) - sig_lin(self.preference - reward)
        # return sig_lin(positive_part) + abs(negative_part)

    # def update_need_after_reward(self, reward):
    #     adjusted_reward = self.reward_function(self.preference) - self.reward_function(self.preference - reward)
    #     self.preference = self.preference - adjusted_reward
        # self.preference = self.preference - reward
        # for i in range(self.num_preference):
        #     self.preference[0, i] = max(self.preference[0, i], -10)

    # def get_total_need(self):
    #     total_need = self.rho_function(self.preference).sum().squeeze()
    #     return total_need

    def take_action(self, environment, action_id):
        selected_action = environment.allactions[action_id].squeeze()  # to device
        self.location[0, :] += selected_action
        at_cost = environment.get_cost(action_id)
        moving_cost = at_cost
        dt = 1 if moving_cost < 1.4 else moving_cost
        environment.update_agent_location_on_map(self)
        f, _ = environment.get_reward()
        satisfaction = (f * self.preference * self.lambda_satisfaction).sum()
        return satisfaction, moving_cost, dt
