from pettingzoo import atari
import importlib
import numpy as np
from supersuit import resize_v0, frame_skip_v0, reshape_v0, max_observation_v0
import gym
import supersuit as ss

def make_env(env_name):
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
    env = max_observation_v0(env, 2)
    env = frame_skip_v0(env, 4)
    # env = InvertColorAgentIndicator(env) # handled by body
    env = resize_v0(env, 84, 84)
    env = reshape_v0(env, (1, 84, 84))
    return env

def InvertColorAgentIndicator(env):
    def modify_obs(obs, agent):
        num_agents = len(env.possible_agents)
        agent_idx = env.possible_agents.index(agent)
        if num_agents == 2:
            if agent_idx == 1:
                rotated_obs = 255 - obs
            else:
                rotated_obs = obs
        elif num_agents == 4:
            rotated_obs = (255*agent_idx)//4 + obs

        indicator = np.zeros((2, )+obs.shape[1:],dtype="uint8")
        indicator[0] = 255 * agent_idx % 2
        indicator[1] = 255 * ((agent_idx+1) // 2) % 2
        return np.concatenate([obs, rotated_obs, indicator], axis=0)
    env = ss.observation_lambda_v0(env, modify_obs)
    env = ss.pad_observations_v0(env)
    return env
