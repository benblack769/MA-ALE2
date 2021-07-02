import copy
import argparse
from all.environments import MultiagentPettingZooEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
import os
import torch
from shared_rainbow import make_rainbow_preset
from independent_rainbow import make_indepedent_rainbow
from shared_ppo import make_ppo_vec
import numpy as np
import random


trainer_types = {
    "shared_rainbow": make_rainbow_preset,
    "independent_rainbow": make_indepedent_rainbow,
    "shared_ppo": make_ppo_vec,
}



def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument("trainer_type", help="Name of the type of training method.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--replay_buffer_size",
        default=1000000,
        type=int,
        help="The size of the replay buffer, if applicable",
    )
    parser.add_argument(
        "--frames", type=int, default=50e6, help="The number of training frames."
    )
    parser.add_argument(
        "--experiment-seed", type=int, default=0, help="The unique id of the experiment run (for running multiple experiments)."
    )

    args = parser.parse_args()

    np.random.seed(args.experiment_seed)
    random.seed(args.experiment_seed)
    torch.manual_seed(args.experiment_seed)

    experiment_name = f"{args.trainer_type}_{args.env}_RB{args.replay_buffer_size}_F{args.frames}"

    experiment, preset, env = trainer_types[args.trainer_type](args.env, args.device, args.replay_buffer_size)

    # run_experiment()
    save_folder = f"checkpoint/{experiment_name}"
    os.makedirs(save_folder)
    num_frames_train = int(args.frames)
    frames_per_save = num_frames_train//100
    for frame in range(0,num_frames_train,frames_per_save):
        experiment.train(frames=frame)
        torch.save(preset, f"{save_folder}/{frame+frames_per_save:09d}.pt")
    # experiment.test(episodes=5)
    experiment._save_model()



if __name__ == "__main__":
    main()
