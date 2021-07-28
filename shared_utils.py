import numpy as np
import torch
# torch.set_num_threads(1)


from all.core import State, StateArray
from all.bodies._body import Body
import torch
import os
from all.bodies.vision import LazyState, TensorDeviceCache

class IndicatorState(State):
    @classmethod
    def from_state(cls, state, frames, to_cache, num_agents, agent_idx):
        state = IndicatorState(state, device=frames[0].device)
        state.to_cache = to_cache
        state.agent_idx = agent_idx
        state.num_agents = num_agents
        state['observation'] = frames
        return state

    def append_agent_to_obs(self, obs):
        if self.num_agents == 2:
            if self.agent_idx == 1:
                rotated_obs = 255 - obs
            else:
                rotated_obs = obs
        elif self.num_agents == 4:
            rotated_obs = (255*self.agent_idx)//4 + obs
        else:
            assert False

        indicator = torch.zeros((2, )+obs.shape[1:],dtype=torch.uint8,device=obs.device)
        indicator[0] = 255 * self.agent_idx % 2
        indicator[1] = 255 * ((self.agent_idx+1) // 2) % 2

        return torch.cat([obs, rotated_obs, indicator], dim=0)

    def __getitem__(self, key):

        if key == 'observation':
            obs = dict.__getitem__(self, key)
            if not torch.is_tensor(obs):
                obs = self.append_agent_to_obs(torch.cat([o for o in obs],dim=0))
            else:
                obs = self.append_agent_to_obs(torch.cat([o.unsqueeze(0) for o in obs],dim=0))

            return obs#torch.cat([obs, indicator], dim=0)
        return super().__getitem__(key)

    def update(self, key, value):
        x = {}
        for k in self.keys():
            if not k == key:
                x[k] = super().__getitem__(k)
        x[key] = value
        state = IndicatorState.from_state(x, x['observation'], self.to_cache, self.num_agents, self.agent_idx)
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
        state = IndicatorState.from_state(x, x['observation'], self.to_cache, self.num_agents, self.agent_idx)
        return state

class IndicatorBody(Body):
    def __init__(self, agent, agent_idx, num_agents):
        super().__init__(agent)
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.to_cache = TensorDeviceCache(max_size=32)

    def process_state(self, state):
        new_state = IndicatorState.from_state(state, dict.__getitem__(state,'observation'), self.to_cache, self.num_agents, self.agent_idx)
        return new_state

class DummyEnv():
    def __init__(self, state_space, action_space, agents):
        self.state_space = state_space
        self.action_space = action_space
        self.agents = agents
