from supersuit.aec_wrappers import ObservationWrapper
from pettingzoo import atari
import importlib
import numpy as np
from supersuit import resize_v0, frame_skip_v0, reshape_v0, max_observation_v0

def make_env(env_name):
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
    env = max_observation_v0(env, 2)
    env = frame_skip_v0(env, 4)
    env = InvertColorAgentIndicator(env)
    env = resize_v0(env, 84, 84)
    env = reshape_v0(env, (1, 84, 84))
    return env

class InvertColorAgentIndicator(ObservationWrapper):
    def _check_wrapper_params(self):
        assert self.observation_spaces[self.possible_agents[0]].high.dtype == np.dtype('uint8')
        return

    def _modify_spaces(self):
        return

    def _modify_observation(self, agent, observation):
        max_num_agents = len(self.possible_agents)
        if max_num_agents == 2:
            if agent == self.possible_agents[1]:
                return self.observation_spaces[agent].high - observation
            else:
                return observation
        elif max_num_agents == 4:
            if agent == self.possible_agents:
                return np.uint8(255//4)+observation
