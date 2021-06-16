from all.environments import GymVectorEnvironment
from all.experiments import ParallelEnvExperiment
from all.presets import atari
from all.agents import Agent
from all.logging import DummyWriter
from all.presets import IndependentMultiagentPreset, Preset
from all.core import State
import torch
from env_utils import make_env
import supersuit as ss
from pettingzoo.utils import to_parallel
from models import impala_features, impala_value_head, impala_policy_head
from env_utils import InvertColorAgentIndicator

def make_vec_env(env_name, device):
    env = make_env(env_name)
    env = ss.black_death_v1(env)
    env = InvertColorAgentIndicator(env)
    env = to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 4, num_cpus=1, base_class='stable_baselines3')
    env = GymVectorEnvironment(env, "env_name", device=device)
    return env


def make_ppo_vec(env_name, device, _):
    venv = make_vec_env(env_name, device)
    preset = atari.ppo.env(venv).hyperparameters(
        n_envs=venv.num_envs,
        n_steps=64,
        minibatches=2,
        epochs=4,
        feature_model_constructor=impala_features,
        value_model_constructor=impala_value_head,
        entropy_loss_scaling=0.001,
        value_loss_scaling=0.1,
        clip_initial=0.5,
        clip_final=0.05,
    ).build()
    experiment = ParallelEnvExperiment(preset, venv)
    return experiment, preset, venv
