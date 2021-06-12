from all.environments import MultiagentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.agents import Agent
from all.logging import DummyWriter
from all.presets import IndependentMultiagentPreset, Preset
from all.core import State
import torch

class SingleEnvAgent(Agent):
    def __init__(self, agent):
        self.agent = agent

    def act(self, state):
        return self.agent.act(State.array([state]))


class ToSingle(Preset):
    def __init__(self, parallel_preset):
        super().__init__(parallel_preset.name, parallel_preset.device, parallel_preset)
        self.parallel_preset = parallel_preset
        assert parallel_preset.n_envs == 1

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        return SingleEnvAgent(self.parallel_preset.agent(writer=writer, train_steps=train_steps))

    def test_agent(self):
        return SingleEnvAgent(self.parallel_preset.test_agent())

def main():
    env_name = "space_invaders_v1"
    device = "cuda"
    env = MultiagentAtariEnv(env_name, device=device)
    presets = {
        'first_0':atari.dqn.env(env.subenvs['first_0']).hyperparameters(replay_buffer_size=10000).build(),
        'second_0':ToSingle(atari.ppo.env(env.subenvs['second_0']).hyperparameters(epochs=5, n_envs=1).build()),
    }
    preset = IndependentMultiagentPreset("atari_experiment", env, presets)

    experiment = MultiagentEnvExperiment(preset, env, write_loss=False, name="independent_"+env_name, save_freq=200000)
    experiment.train(frames=1000000)
    torch.save(preset,"trained_model.th")
    experiment.test(3)

if __name__ == "__main__":
    main()
