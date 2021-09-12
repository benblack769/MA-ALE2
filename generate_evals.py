from all_envs import all_environments, builtin_envs

four_p_envs = {
"warlords_v2",
"quadrapong_v3",
"volleyball_pong_v",
"foozpong_v2",
}

experiment_configs = [
    ("shared_rainbow", 1000000),
    ("shared_ppo", 1),
]

num_frames_train = 50000000
frames_per_save = num_frames_train//100

eval_frames = 125000
base_folder = "experiments/ppo_train"

num_experiments = 5

vs_builtin = True

device = "--device=cpu"

if vs_builtin:
    all_environments = {name:env for name, env in all_environments.items() if name in builtin_envs}

def make_name(trainer, env, buf_size, experiment):
    return f"{trainer}_{env}_RB{buf_size}_F{num_frames_train}_S{experiment}"

agent_2p_list = ["first_0", "second_0"]
agent_4p_list = agent_2p_list + ["third_0", "fourth_0"]
for env in all_environments:
    for experiment in range(num_experiments):
        for trainer, buf_size in experiment_configs:
            checkpoint_folder = f"{base_folder}/{make_name(trainer, env, buf_size, experiment)}"
            agent_list = agent_4p_list if env in four_p_envs else agent_2p_list
            for checkpoint in range(frames_per_save, num_frames_train, frames_per_save):
                vs_random_list = ["", "--vs-random"] if not vs_builtin else [""]
                for vs_random in vs_random_list:
                    # if vs_random:
                    #     for agent in agent_list:
                    #         frames = eval_frames
                    #         print(f"workon main_env && python experiment_eval.py {env} {checkpoint:09d} {checkpoint_folder} --frames={frames} --agent={agent} {vs_random}")
                    # else:
                    agent = "first_0"
                    vs_builtin_str = "--vs-builtin" if vs_builtin else ''
                    frames = eval_frames
                    print(f"execute_remote --copy-forward *.py {checkpoint_folder}/{checkpoint:09d}.pt --copy-backwards out.txt --verbose '. /opt/conda/etc/profile.d/conda.sh && conda activate && python experiment_eval.py {env} {checkpoint:09d} {checkpoint_folder} --frames={frames} --agent={agent} {vs_random} {vs_builtin_str} {device}'")
        # frames = eval_frames*4
        # print(f"workon main_env && python experiment_eval.py {env} {frames_per_save} ~/job_results/ --frames={frames} --agent={agent} --vs-random --agent-random")
