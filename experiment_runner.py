import copy
import argparse
from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
import os
import torch
from shared_rainbow import make_rainbow_preset

def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--replay_buffer_size",
        default=1000000,
        help="The size of the replay buffer, if applicable",
    )
    parser.add_argument(
        "--frames", type=int, default=50e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment."
    )
    args = parser.parse_args()

    preset, env = make_rainbow_preset(args.env, args.device)

    experiment = MultiagentEnvExperiment(
        preset,
        env,
        write_loss=False,
        render=args.render,
    )
    # run_experiment()
    os.mkdir("checkpoint")
    num_frames_train = int(args.frames)
    frames_per_save = 200000
    for frame in range(0,num_frames_train,frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"checkpoint/{frame+frames_per_save:08d}.pt")
    # experiment.test(episodes=5)
    experiment._save_model()



if __name__ == "__main__":
    main()
