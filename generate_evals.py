from all_envs import all_environments

four_p_envs = {
"warlords_v2",
"quadrapong_v3",
"volleyball_pong_v",
"foozpong_v2",
}

experiment_configs = [
    ("shared_ppo", 1000000),
    # ("shared_rainbow", 100000),
    # ("independent_rainbow", 1000000),
    # ("independent_rainbow", 100000),
]

num_frames_train = 20000000
frames_per_save = num_frames_train//100

eval_frames = 125000
base_folder = "experiments/ppo_long"

def make_name(trainer, env, buf_size):
    return f"{trainer}_{env}_RB{buf_size}_F{num_frames_train}"

agent_2p_list = ["first_0", "second_0"]
agent_4p_list = agent_2p_list + ["third_0", "fourth_0"]
for env in all_environments:
    for trainer, buf_size in experiment_configs:
        checkpoint_folder = f"{base_folder}/{make_name(trainer, env, buf_size)}"
        agent_list = agent_4p_list if env in four_p_envs else agent_2p_list
        for checkpoint in range(frames_per_save, num_frames_train, frames_per_save):
            for vs_random in ["", "--vs-random"]:
                # if vs_random:
                #     for agent in agent_list:
                #         frames = eval_frames
                #         print(f"workon main_env && python experiment_eval.py {env} {checkpoint:09d} {checkpoint_folder} --frames={frames} --agent={agent} {vs_random}")
                # else:
                agent = "first_0"
                frames = eval_frames
                print(f"execute_remote --copy-forward *.py {checkpoint_folder}/{checkpoint:09d}.pt --copy-backwards out.txt --verbose 'workon main_env && python experiment_eval.py {env} {checkpoint:09d} {checkpoint_folder} --frames={frames} --agent={agent} {vs_random}'")
    # frames = eval_frames*4
    # print(f"workon main_env && python experiment_eval.py {env} {frames_per_save} ~/job_results/ --frames={frames} --agent={agent} --vs-random --agent-random")
