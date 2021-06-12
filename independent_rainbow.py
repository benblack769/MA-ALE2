from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.agents import Agent
from all.logging import DummyWriter
from all.presets import IndependentMultiagentPreset, Preset
from all.core import State
import torch
from env_utils import make_env


def make_indepedent_rainbow(env_name, device, replay_buffer_size):
    env = make_env(env_name)
    env = MultiagentPettingZooEnv(env, env_name, device=device)
    presets = {
        agent: atari.rainbow.env(env.subenvs['first_0']).hyperparameters(replay_buffer_size=replay_buffer_size).build()
            for agent in env.agents
    }
    preset = IndependentMultiagentPreset("atari_experiment", env, presets)
    return preset, env
