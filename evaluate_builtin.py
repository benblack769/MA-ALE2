import argparse
import copy
import os
import numpy as np
import random
import subprocess
import gym
import all
from all.approximation import QDist, FixedTarget
from all.agents import Rainbow, RainbowTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from all.presets.atari.models import nature_rainbow
from all.presets.preset import Preset
from all.presets.builder import PresetBuilder
from all.agents.independent import IndependentMultiagent
from all.environments import MultiagentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.core import State
from timeit import default_timer as timer
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from supersuit import color_reduction_v0, frame_stack_v1, max_observation_v0, frame_skip_v0, resize_v0
from gym.wrappers import AtariPreprocessing
from all.environments.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    LifeLostEnv,
)

from all.bodies.vision import FrameStack

from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 6.25e-5,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 32,
    "update_frequency": 4,
    "target_update_frequency": 1000,
    # Replay buffer settings
    "replay_start_size": 80000,
    "replay_buffer_size": 1000000,
    # Explicit exploration
    "initial_exploration": 0.02,
    "final_exploration": 0.,
    "test_exploration": 0.001,
    # Prioritized replay settings
    "alpha": 0.5,
    "beta": 0.5,
    # Multi-step learning
    "n_steps": 3,
    # Distributional RL
    "atoms": 51,
    "v_min": -10,
    "v_max": 10,
    # Noisy Nets
    "sigma": 0.5,
    # Model construction
    "model_constructor": nature_rainbow
}

class RainbowAtariPreset(Preset):
    """
    Rainbow DQN Atari Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        device (torch.device, optional): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        alpha (float): Amount of prioritization in the prioritized experience replay buffer.
            (0 = no prioritization, 1 = full prioritization)
        beta (float): The strength of the importance sampling correction for prioritized experience replay.
            (0 = no correction, 1 = full correction)
        n_steps (int): The number of steps for n-step Q-learning.
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return correspodning to the larget atom.
        sigma (float): Initial noisy network noise.
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, name, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__(env, name, hyperparameters)
        self.model = hyperparameters['model_constructor'](env, frames=8, atoms=hyperparameters["atoms"], sigma=hyperparameters["sigma"]).to(device)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device
        self.name = name
        self.agent_names = env.agents

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q_dist = QDist(
            self.model,
            optimizer,
            self.n_actions,
            self.hyperparameters['atoms'],
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer,
        )

        replay_buffer = NStepReplayBuffer(
            self.hyperparameters['n_steps'],
            self.hyperparameters['discount_factor'],
            PrioritizedReplayBuffer(
                self.hyperparameters['replay_buffer_size'],
                alpha=self.hyperparameters['alpha'],
                beta=self.hyperparameters['beta'],
                device=self.device,
                store_device="cpu"
            )
        )

        def make_agent(agent_id):
            return DeepmindAtariBody(
                IndicatorBody(
                    Rainbow(
                        q_dist,
                        replay_buffer,
                        exploration=LinearScheduler(
                            self.hyperparameters['initial_exploration'],
                            self.hyperparameters['final_exploration'],
                            0,
                            train_steps - self.hyperparameters['replay_start_size'],
                            name="exploration",
                            writer=writer
                        ),
                        discount_factor=self.hyperparameters['discount_factor'] ** self.hyperparameters["n_steps"],
                        minibatch_size=self.hyperparameters['minibatch_size'],
                        replay_start_size=self.hyperparameters['replay_start_size'],
                        update_frequency=self.hyperparameters['update_frequency'],
                        writer=writer,
                    ),
                    self.agent_names.index(agent_id),
                    len(self.agent_names)
                ),
                lazy_frames=True,
                episodic_lives=True
            )

        return IndependentMultiagent({
            agent_id : make_agent(agent_id)
            for agent_id in self.agent_names
        })

    def test_agent(self):
        q_dist = QDist(
            copy.deepcopy(self.model),
            None,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        def make_agent():
            return DeepmindAtariBody(RainbowTestAgent(q_dist, self.n_actions, self.hyperparameters["test_exploration"]))
        return IndependentMultiagent({
            agent_id : make_agent()
            for agent_id in self.agent_names
        })

rainbow = PresetBuilder('rainbow', default_hyperparameters, RainbowAtariPreset)

from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
import numpy as np
# from all.experiment import run_
from supersuit.aec_wrappers import ObservationWrapper

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

from all.core import State, StateArray
from all.bodies._body import Body
import torch
import os
from all.bodies.vision import LazyState, TensorDeviceCache

class IndicatorState(State):
    @classmethod
    def from_state(cls, state, frames, to_cache, agent_idx):
        state = IndicatorState(state, device=frames[0].device)
        state.to_cache = to_cache
        state.agent_idx = agent_idx
        state['observation'] = frames
        return state

    def __getitem__(self, key):
        if key == 'observation':
            v = dict.__getitem__(self, key)
            if torch.is_tensor(v):
                return v
            obs = torch.cat(dict.__getitem__(self, key), dim=0)
            indicator = torch.zeros_like(obs)
            indicator[self.agent_idx] = 255
            return torch.cat([obs, indicator], dim=0)

        return super().__getitem__(key)

    def update(self, key, value):
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        state = IndicatorState(x, device=self.device)
        state.to_cache = self.to_cache
        state.agent_idx = self.agent_idx
        return state

    def to(self, device):
        if device == self.device:
            return self
        x = {}
        for key, value in self.items():
            if key == 'observation':
                x[key] = [self.to_cache.convert(v, device) for v in value]
                # x[key] = [v.to(device) for v in value]#torch.cat(value,axis=0).to(device)
            elif torch.is_tensor(value):
                x[key] = value.to(device)
            else:
                x[key] = value
        state = IndicatorState.from_state(x, x['observation'], self.to_cache, self.agent_idx)
        return state

class IndicatorBody(Body):
    def __init__(self, agent, agent_idx, num_agents):
        super().__init__(agent)
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.to_cache = TensorDeviceCache(max_size=32)

    def process_state(self, state):
        return IndicatorState.from_state(state, dict.__getitem__(state,'observation'), self.to_cache, self.agent_idx)

class DummyEnv():
    def __init__(self, state_space, action_space, agents):
        self.state_space = state_space
        self.action_space = action_space
        self.agents = agents

def generate_episode_gifs(env, _agent, max_frames, save_dir, side="first_0"):
    # initialize the episode
    observation = env.reset()
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    done = False
    while not done:
        #print(_agent.agents)
        obs = State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8)
        print(obs)
        obs["agent"] = "first_0" 
        action = _agent.act(obs)
        observation, reward, done, info = env.step(action)
        print(observation.shape)
        if reward != 0.0:
            print(reward)
       # obs = env.render(mode='rgb_array')
        if not prev_obs or not np.equal(observation, prev_obs).all():
            im = Image.fromarray(observation)
            im.save(f"{save_dir}{str(frame_idx).zfill(4)}.png")

            frame_idx += 1
            if frame_idx >= max_frames:
                break

def test_single_episode(env, _agent, generate_gif_callback=None, side="first_0"):
    # initialize the episode
    observation = env.reset()
    returns = 0
    num_steps = 0
    frame_idx = 0
    prev_obs = None
    print(side)

    # loop until the episode is finished
    done = False
    while not done:
        print(_agent.agents)
        ob = State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8)
        print(ob)
        ob["agent"] = "first_0" 
        action = _agent.act(ob)
        observation, reward, done, info = env.step(action)
        returns += reward
        num_steps += 1

    return returns, num_steps

def test_independent(env, agent, frames, side="first_0"):
    returns = []
    num_steps = 0
    while num_steps < frames:
        episode_return, ep_steps = test_single_episode(env, agent, side=side)
        returns.append(episode_return)
        num_steps += ep_steps
        print(num_steps)
        # self._log_test_episode(episode, episode_return)
    return returns

def returns_agent(returns, agent):
    print(returns)
    print(np.mean(returns))
    return np.mean(returns)

from supersuit.aec_wrappers import ObservationWrapper

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

class InvertColorWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super(InvertColorWrapper, self).__init__(env)
        assert self.observation_space.high.dtype == np.dtype('uint8')

    def observation(self, observation):
        return self.observation_space.high - observation

def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument("checkpoint", help="Name of the checkpoint (e.g. pong_v1).")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=100000, help="The number of training frames."
    )
    parser.add_argument(
        "--agent", type=str, default="first_0", help="Agent to print out value."
    )
    parser.add_argument(
        "--generate-gif", action="store_true", help="Agent to print out value."
    )
    # parser.add_argument(
    #     "--render", type=bool, default=False, help="Render the environment."
    # )
    args = parser.parse_args()

    # self._writer = ExperimentWriter(self, multiagent.__name__, env.name, loss=write_loss)
    # self._writers = {
    #     agent : ExperimentWriter(self, "{}_{}".format(multiagent.__name__, agent), env.name, loss=write_loss)
    #     for agent in env.agents
    # }
    frame_skip = 1 if args.generate_gif else 4
    env = gym.make(args.env, full_action_space=True, obs_type="image")
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = WarpFrame(env)
    #if args.agent == "second_0":
    #    env = InvertColorWrapper(env)
    #env = FrameStack(env, 4)
    #env = InvertColorWrapper(env, agent=args.agent)
    env.reset()

    # base_builder = getattr(atari, agent_name)()
    #preset = torch.load(os.path.join("./checkpoints/latest_ind_atari_checkpoints/" + args.checkpoint + "_final_checkpoint.th"))
    preset = torch.load(os.path.join("./checkpoint/09000000.pt"))
    agent = preset.test_agent()
    print(agent)

    if not args.generate_gif:
        returns = test_independent(env, agent, args.frames, side=args.agent)
        agent_names = ["first_0", "second_0", "third_0", "fourth_0"]
        with open("./outfiles/" + args.checkpoint + "_" + args.agent + ".txt",'w') as out:
            out.write(f"Environment: {args.env}\n")
            out.write(f"Checkpoint Name: {args.checkpoint}\n")
            out.write(f"Agent: {args.agent}\n")
            out.write(f"Returns: {str(returns)}\n")
            out.write(f"Average return: {str(np.mean(returns))}\n")
            out.write(f"Evaluation frames: {args.frames}\n")

    else:
        name = f"{args.env}_{args.agent}"
        folder = f"frames/{name}/"
        os.makedirs(folder,exist_ok=True)
        os.makedirs("gifs",exist_ok=True)
        generate_episode_gifs(env, agent, args.frames, folder, side=args.agent)

        ffmpeg_command = [
            "ffmpeg",
            "-framerate", "60",
            "-i", f"frames/{name}/%04d.png",
            "-vcodec", "libx264",
            "-crf", "1",
            "-pix_fmt", "yuv420p",
            f"gifs/{name}.mp4"
        ]
        # ffmpeg_command = [
        #     "convert",
        #     '-delay','1x120',
        #      f"frames/{name}/*.png",
        #     f"gifs/{name}.gif"
        # ]
        subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    main()
