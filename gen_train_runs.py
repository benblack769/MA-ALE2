from all_envs import all_environments

experiment_configs = [
    ("shared_rainbow", 1000000),
    ("shared_rainbow", 100000),
    ("independent_rainbow", 1000000),
    ("independent_rainbow", 100000),
]

num_frames = 10000000
num_experiments = 1
for env_name in sorted(all_environments):
    for algo_name, replay_size in experiment_configs:
        for experiment in range(num_experiments):
            print(f"workon main_env && python experiment_runner.py {env_name} {algo_name} --replay_buffer_size={replay_size} --frames={num_frames}")
