import math

import os
from sb3_contrib.ppo_mask import MaskablePPO
import torch
import numpy as np
import itertools
from Environment import Environment
import matplotlib.pyplot as plt


def get_predefined_parameters(params, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * params.OBJECT_TYPE_NUM
    elif param_name == 'all_object_rewards':
        # all_param = [[0, 4, 8, 12, 16, 20]] * num_object
        param_range = params.ENVIRONMENT_OBJECT_REWARD_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1], num=min(param_range[1] - param_range[0] + 1, 4),
                                               dtype=int), axis=0).tolist() * params.OBJECT_TYPE_NUM
    elif param_name == 'all_mental_states_change':
        # all_param = [[0, 1, 2, 3, 4, 5]] * num_object
        param_range = params.MENTAL_STATES_SLOPE_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1],
                                               num=min(param_range[1] - param_range[0] + 1, 4), dtype=int),
                                   axis=0).tolist() * params.OBJECT_TYPE_NUM
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** params.OBJECT_TYPE_NUM
    param_batch = []
    for i, ns in enumerate(itertools.product(*all_param)):
        param_batch.append(list(ns))
    return param_batch


class Test:
    def __init__(self, utils):
        self.debug = False
        self.params = utils.params
        self.res_folder = utils.res_folder
        self.model = self.load_model()
        self.height = utils.params.HEIGHT
        self.width = utils.params.WIDTH
        self.object_type_num = utils.params.OBJECT_TYPE_NUM
        self._all_actions = np.array([[0, 0],
                                      [1, 0], [-1, 0], [0, 1], [0, -1],
                                      [1, 1], [-1, -1], [-1, 1], [1, -1]])

        self.all_mental_states = get_predefined_parameters(self.params, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.params, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.params, 'all_mental_states_change')

        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay

    def get_figure_title(self, mental_states):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', mental_states[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', mental_states[i])
        return title

    def next_agent_and_environment(self):
        for object_reward in self.all_object_rewards:
            for mental_state_slope in self.all_mental_states_change:
                environment = Environment(self.params, ['few', 'many'])

                for subplot_id, mental_state in enumerate(self.all_mental_states):
                    for i in range(self.height):
                        for j in range(self.width):
                            action_mask, flat_env, object_locations, each_type_object_num = environment.init_environment_for_test(
                                [i, j],
                                mental_state,
                                mental_state_slope,
                                object_reward)
                            env_parameters = [mental_state, mental_state_slope, object_reward]
                            yield action_mask, flat_env, [i,
                                                  j], object_locations, each_type_object_num, env_parameters, subplot_id

    def get_goal_directed_actions(self):
        fig, ax = None, None
        which_goal = None
        row_num = 5
        col_num = 5
        created_subplot = np.zeros((row_num * col_num,), dtype=bool)
        for setting_id, outputs in enumerate(self.next_agent_and_environment()):
            action_mask = outputs[0]
            flat_environment = outputs[1]
            agent_location = outputs[2]
            object_locations = outputs[3]
            each_type_object_num = outputs[4]
            env_parameters = outputs[5]  # [mental_state, mental_state_slope, object_reward]
            subplot_id = outputs[6]

            if setting_id % (col_num * row_num * self.width * self.height) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
            if setting_id % (self.height * self.width) == 0:
                which_goal = np.empty((self.height, self.width), dtype=str)

            r = subplot_id // col_num
            c = subplot_id % col_num

            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()

            # shape_map = self.get_object_shape_dictionary(object_locations, agent_location, each_type_object_num)

            with torch.no_grad():
                action = self.model.predict(observation=flat_environment,
                                            deterministic=True,
                                            action_masks=action_mask)[0]
                step = self._all_actions[action]
            scale = .2
            ax[r, c].arrow(x=agent_location[1], y=agent_location[0],
                           dx=step[1]*scale, dy=step[0]*scale,
                           head_width=.2, length_includes_head=True)

            if self.debug or agent_location[0] == self.height - 1 and agent_location[1] == self.width - 1:
                ax[r, c].set_title(self.get_figure_title(env_parameters[0]), fontsize=10)
                for obj_type in range(self.object_type_num):
                    at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
                    for obj in range(each_type_object_num[obj_type]):

                        ax[r, c].scatter(at_type_object_locations[obj, 1:][1],
                                         at_type_object_locations[obj, 1:][0],
                                         marker=self.goal_shape_options[obj],
                                         s=200,
                                         edgecolor=self.color_options[obj_type],
                                         facecolor='none')
                ax[r, c].tick_params(length=0)
                ax[r, c].set(adjustable='box')
            if self.debug or (setting_id + 1) % (col_num * row_num * self.width * self.height) == 0:
                plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
                fig.savefig('{0}/slope_{1}-{2}_or_{3}-{4}.png'.format(self.res_folder,
                                                                      env_parameters[1][0],
                                                                      env_parameters[1][1],
                                                                      env_parameters[2][0],
                                                                      env_parameters[2][1]))
                plt.close()

    def load_model(self):
        model_path = os.path.join(self.res_folder, 'model.zip')
        return MaskablePPO.load(path=model_path)
