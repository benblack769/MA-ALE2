import numpy as np
import torch
torch.set_num_threads(1)


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
            obs = dict.__getitem__(self, key)
            if not torch.is_tensor(obs):
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
        new_state = IndicatorState.from_state(state, dict.__getitem__(state,'observation'), self.to_cache, self.agent_idx)
        return new_state

class DummyEnv():
    def __init__(self, state_space, action_space, agents):
        self.state_space = state_space
        self.action_space = action_space
        self.agents = agents
