from all_envs import all_environments

experiment_configs = [
    # ("shared_ppo", 1),
    ("shared_rainbow", 1000000),
    # ("independent_rainbow", 1000000),
    # ("independent_rainbow", 100000),
]

num_frames = 50000000
num_experiments = 5
for env_name in sorted(all_environments):
    for exp_num in range(num_experiments):
        for algo_name, replay_size in experiment_configs:
            print(f"workon main_env && python experiment_train.py {env_name} {algo_name} --replay_buffer_size={replay_size} --frames={num_frames} --experiment-seed={exp_num}")
