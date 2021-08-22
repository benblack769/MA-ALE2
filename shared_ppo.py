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
from models import impala_features, impala_value_head, impala_policy_head, nature_features
from env_utils import InvertColorAgentIndicator
from all.bodies import DeepmindAtariBody
from nfsp_models import ImpalaCNNLarge
from all import nn


def make_vec_env(env_name, device):
    import importlib
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = InvertColorAgentIndicator(env) # handled by body
    env = ss.resize_v0(env, 84, 84)
    env = ss.reshape_v0(env, (1, 84, 84))
    env = ss.black_death_v2(env)
    env = InvertColorAgentIndicator(env)
    # env = to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 32, num_cpus=8, base_class='stable_baselines3')
    env = GymVectorEnvironment(env, "env_name", device=device)
    return env


def nat_features():
    return nature_features(16)


def make_ppo_vec(env_name, device, _):
    venv = make_vec_env(env_name, device)
    preset = atari.ppo.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        n_steps=32,
        minibatches=8,
        epochs=4,
        feature_model_constructor=nat_features,
        # value_model_constructor=impala_value_head,
        # policy_model_constructor=impala_policy_head,
        entropy_loss_scaling=0.001,
        value_loss_scaling=0.1,
        clip_initial=0.5,
        clip_final=0.05,
    ).build()
    # base_agent = preset.agent.agent.agent
    # preset = DeepmindAtariBody(base_agent, lazy_frames=True, episodic_lives=False, clip_rewards=True,)
    # print(base_agent)

    experiment = ParallelEnvExperiment(preset, venv)
    return experiment, preset, venv


def impala_value_head():
    return nn.Linear(256, 1)


def impala_policy_head(env):
    return nn.Linear0(256, env.action_space.n)


def largenet():
    largenet = ImpalaCNNLarge(16, 18, nn.Linear, (84, 84), model_size=2)
    return largenet


def make_ppo_vec_largenet(env_name, device, _):
    venv = make_vec_env(env_name, device)
    n_steps = (256*32*2) // venv.num_envs
    preset = atari.ppo.env(venv).device(device).hyperparameters(
        n_envs=venv.num_envs,
        n_steps=n_steps,
        minibatches=32,
        epochs=4,
        feature_model_constructor=largenet,
        value_model_constructor=impala_value_head,
        policy_model_constructor=impala_policy_head,
        entropy_loss_scaling=0.001,
        value_loss_scaling=0.1,
        clip_initial=0.5,
        clip_final=0.05,
    ).build()
    # base_agent = preset.agent.agent.agent
    # preset = DeepmindAtariBody(base_agent, lazy_frames=True, episodic_lives=False, clip_rewards=True,)
    # print(base_agent)

    experiment = ParallelEnvExperiment(preset, venv)
    return experiment, preset, venv
