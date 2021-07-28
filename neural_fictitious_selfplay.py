import pettingzoo
import supersuit as ss
from all import nn
import torch
import random
from tianshou.data import Batch
import reverb
import multiprocessing

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
    proc = multiprocessing.Process(target=start_server, args=(max_size, agents))


def add_element(traj_writer, agent, obs, action, reward, done):
    writer.create_item(
      table='my_table',
      priority=1.0,
      trajectory={
          'a': writer.history['a'][:],
          'b': writer.history['b'][:],
      })

def make_env(env_name):
    env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).env(obs_type='grayscale_image')
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = InvertColorAgentIndicator(env) # handled by body
    env = ss.resize_v0(env, 84, 84)
    env = ss.reshape_v0(env, (1, 84, 84))
    return env


class ReservourBuffer:
    def __init__(self, buffer, size):
        self.buffer = buffer
        self.index = 0
        self.fill_index = 0
        self.buf_size = size

    def add(self, obs):
        if self.fill_index < self.buf_size:
            self.buffer[self.fill_index] = obs
            self.fill_index += 1
        else:
            new_idx = random.randint(0, self.index)
            if new_idx < self.buf_size:
                self.buffer[new_idx] = obs
        self.index += 1

    def sample(self, batch_size):
        idxs = torch.randint(self.fill_index, size=batch_size)
        return self.buffer[idxs]


class QueueBuffer:
    def __init__(self, buffer, size):
        self.buffer = buffer
        self.index = 0
        self.fill_index = 0
        self.buf_size = size

    def add(self, obs):
        if self.fill_index < self.buf_size:
            self.fill_index += 1

        self.buffer[self.index] = obs
        self.index += 1
        if self.index >= self.buf_size:
            self.index = 0

    def sample(self, batch_size):
        idxs = torch.randint(self.fill_index, size=batch_size)
        return self.buffer[idxs]


#TODO: implement anticipory dynamics


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
    def __init__(self, batch_size):
        self.model = rl_model()
        self.batch_size = batch_size
        buf_size = 500000
        img_shape = (4, 84, 84)
        buf = Batch({
            "observation": torch.zeros((buf_size,)+img_shape,dtype=torch.uint8),
            "done": torch.zeros((buf_size,),device=store_device),
            "reward": torch.zeros((buf_size,),device=store_device),
            "action": torch.zeros((buf_size,),dtype=torch.int64),
        })
        self.queue_buffer = QueueBuffer(buf, 500000)

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
