import pettingzoo
import supersuit as ss
from all import nn
import torch
import random
from tianshou.data import Batch
import reverb
import multiprocessing
import importlib
import psutil
import atexit
import numpy as np

def start_server(max_size, agents):
    all_tables = []
    for agent in agents:
        all_tables.append(reverb.Table(
            name='reservour_buffer_' + agent,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Uniform(),
            max_size=max_size,
            rate_limiter=reverb.rate_limiters.MinSize(1)
        ))
        all_tables.append(reverb.Table(
            name='queue_buffer_' + agent,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_size,
            rate_limiter=reverb.rate_limiters.MinSize(1)
        ))

    server = reverb.Server(tables=all_tables, port=8000)


def run_server(max_size, agents):
    return multiprocessing.Process(target=start_server, args=(max_size, agents))

def cleaup_processes():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        child.terminate()

atexit.register(cleaup_processes)


def add_element(traj_writer, agent, obs, action, reward, done):
    writer.create_item(
      table='my_table',
      priority=1.0,
      trajectory={
          'a': writer.history['a'][:],
          'b': writer.history['b'][:],
      })


def make_env(env_name):
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = InvertColorAgentIndicator(env) # handled by body
    env = ss.resize_v0(env, 84, 84)
    env = ss.reshape_v0(env, (1, 84, 84))
    return env


def make_vec_env(base_env, num_envs, num_cpus):
    base_env = ss.black_death_v1(base_env)
    env = ss.pettingzoo_env_to_vec_env_v0(base_env)
    env = ss.concat_vec_envs_v0(env, 4, num_cpus=2, base_class='stable_baselines3')
    return env, base_env.possible_agents


class RandomAgent:
    def act(self, obs):
        assert obs.obs.shape[1:] == (1, 84, 84)

        return torch.randint(high=18, size=(obs.obs.shape[0],))


def apply_all_in_batch(batch, op):
    batch_dict = {}
    for name, tensor in batch.__dict__.items():
        batch_dict[name] = op(tensor)
    return Batch(batch_dict)


class AgentDistributor:
    '''
    Maps multi-agent flattened vector env layout onto multiple agents.
    '''
    def __init__(self, agents):
        self.agents = agents
        self.num_agents = len(self.agents)

    def act(self, all_observations):
        reshaped_obs = apply_all_in_batch(all_observations, lambda obs: obs.view((obs.shape[0]//self.num_agents,self.num_agents,)+obs.shape[1:]))
        agent_first_obs = apply_all_in_batch(reshaped_obs, lambda obs: torch.swapaxes(obs,0,1))
        agent_actions = []
        for i, agent in enumerate(self.agents):
            actions = agent.act(agent_first_obs[i])
            agent_actions.append(actions)
        vec_env_actions = torch.stack(agent_actions,axis=1)
        assert vec_env_actions.shape[:2] == reshaped_obs.obs.shape[:2]
        vec_env_actions = vec_env_actions.view((all_observations.obs.shape[0],)+vec_env_actions.shape[2:])
        return vec_env_actions


def obs_are_zeros(batched_obs):
    '''
    check if observations are zeros (mean black death, means they should not be learned)
    '''
    return torch.all(torch.eq(batched_obs.view(batched_obs.shape[0], -1), 0), dim=1).type(torch.uint8)


def make_obs_batch_from_env(observations, rewards, dones, device):
    torch_obs = torch.tensor(observations, device=device)
    torch_rews = torch.tensor(rewards, device=device, dtype=torch.float32)
    torch_dones = torch.tensor(dones, device=device, dtype=torch.uint8)
    torch_mask = 1 - (torch_dones | obs_are_zeros(torch_obs))
    return Batch({
        "obs": torch_obs,
        "rewards": torch_rews,
        "dones": torch_dones,
        "mask": torch_mask,
    })


def env_loop(env_name, num_steps, num_envs, device="cpu"):
    base_env = make_env(env_name)
    v_env, agent_names = make_vec_env(base_env, num_envs, num_cpus=4)
    num_agents = len(agent_names)
    agents = AgentDistributor([RandomAgent() for i in range(num_agents)])

    observations = v_env.reset()
    rewards = np.zeros(len(observations))
    dones = np.zeros(len(observations))
    for i in range(num_steps):
        batched_data = make_obs_batch_from_env(observations, rewards, dones, device)
        actions = agents.act(batched_data)
        observations, rewards, dones, infos = v_env.step(actions)


# env_loop("space_invaders_v1",100,4)


def rl_model(frames=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.Linear(3136, hidden),
                nn.ReLU(),
                nn.Linear0(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.Linear(3136, hidden),
                nn.ReLU(),
                nn.Linear0(
                    hidden,
                    18 * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )


def sl_model(frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, 18),
    )


class RLAgent:
    def __init__(self, q_model, batch_size):
        self.model = q_model
        self.batch_size = batch_size
        buf_size = 500000
        img_shape = (4, 84, 84)
        buf = Batch({
            "observation": torch.zeros((buf_size,)+img_shape,dtype=torch.uint8),
            "done": torch.zeros((buf_size,),device=store_device),
            "reward": torch.zeros((buf_size,),device=store_device),
            "action": torch.zeros((buf_size,),dtype=torch.int64),
        })
        

    def act(self, obs_batch):
        pass


class SLAgent:
    def __init__(self, batch_size):
        self.mode = sl_model()
        self.batch_size = batch_size
        buf_size = 1000000
        img_shape = (4, 84, 84)
        buf = Batch({
            "observation": torch.zeros((buf_size,)+img_shape,dtype=torch.uint8),
            "action": torch.zeros((buf_size,),dtype=torch.int64),
        })

    def train(self):
        train_data = buf.sample(self.batch_size)

    def act(self, obs_batch):
        pass

class CombinedAgent:
    def __init__(self, num_envs, prob_select_rl):
        self.rl_agent = RLAgent()
        self.sl_agent = SLAgent()
        self.num_envs = num_envs

        self.env_rl_agent = torch.zeros(num_envs, dtype=bool)
        self.random = np.random.RandomState(42)
        self.prob_select_rl = prob_select_rl
        for i in range(num_envs):
            self.remap_env(i)

    def remap_env(self, env_id):
        self.env_rl_agent[env_id] = self.random.random() < self.prob_select_rl

    def act(self, obs_batch):
        sl_data = obs_batch[~self.env_rl_agent]
        rl_data = obs_batch[self.env_rl_agent]
        rl_actions = self.rl_agent.act(rl_data)
        sl_actions = self.sl_agent.act(sl_data)

        all_actions = torch.zeros(self.num_envs, dtype=int)
        all_actions[~self.env_sl_agent] = sl_actions
        all_actions[self.env_rl_agent] = rl_actions

        return all_actions





















#
