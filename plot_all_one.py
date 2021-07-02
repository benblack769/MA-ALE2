import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas
import os
import scipy
from scipy import signal
import sys

from all_envs import all_environments

def get_env_name(experiment_name):
    for env_name in all_environments:
        if env_name in experiment_name:
            return env_name
    raise AssertionError(f"{experiment_name} does not contain any environment name: {list(sorted(all_environments))}")
    # # env_name = env_name[:env_name.rfind("_")]
    # if 'boxing' in env_name:
    #     game_env = 'Boxing'
    # elif 'combat_plane' in env_name:
    #     game_env = 'Combat Plane'
    # elif 'combat_tank' in env_name:
    #     game_env = 'Combat Tank'
    # elif 'double_dunk' in env_name:
    #     game_env = 'Double Dunk'
    # elif 'entombed_competitive' in env_name:
    #     game_env = 'Entombed Competitive'
    # elif 'entombed_cooperative' in env_name:
    #     game_env = 'Entombed Cooperative'
    # elif 'flag_capture' in env_name:
    #     game_env = 'Flag Capture'
    # elif 'ice_hockey' in env_name:
    #     game_env = 'Ice Hockey'
    # elif 'joust' in env_name:
    #     game_env = 'Joust'
    # elif 'mario_bros' in env_name:
    #     game_env = 'Mario Bros'
    # elif 'maze_craze' in env_name:
    #     game_env = 'Maze Craze'
    # elif 'othello' in env_name:
    #     game_env = 'Othello'
    # elif 'basketball_pong' in env_name:
    #     game_env = 'Basketball Pong'
    # elif 'pong' in env_name:
    #     game_env = 'Pong'
    # elif 'foozpong' in env_name:
    #     game_env = 'Foozpong'
    # elif 'quadrapong' in env_name:
    #     game_env = 'Quadrapong'
    # elif 'volleyball_pong' in env_name:
    #     game_env = 'Volleyball Pong'
    # elif 'space_invaders' in env_name:
    #     game_env = 'Space Invaders'
    # elif 'space_war' in env_name:
    #     game_env = 'Space War'
    # elif 'surround' in env_name:
    #     game_env = 'Surround'
    # elif 'tennis' in env_name:
    #     game_env = 'Tennis'
    # elif 'video_checkers' in env_name:
    #     game_env = 'Video Checkers'
    # elif 'warlords' in env_name:
    #     game_env = 'Warlords'
    # elif 'wizard_of_wor' in env_name:
    #     game_env = 'Wizard of Wor'
    # else:
    #     raise RuntimeError(f"{env_name} not found")
    # return game_env

def get_exp_label(exp):
    start = exp.find("/")+1
    end = exp.rfind("_")
    return exp[start:end]

def main():

    assert len(sys.argv) == 2

    csv_name = sys.argv[1]

    matplotlib.use("pgf")
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 6,
        "legend.fontsize": 5,
        "ytick.labelsize": 4,
        "text.usetex": True,
        "pgf.rcfonts": False
    });

    plt.figure(figsize=(2.65*3*1.0, 1.5*7*1.0))

    csv_data = pandas.read_csv(csv_name)
    random_data = pandas.read_csv("rand_data.csv")
    random_data = random_data[random_data['vs_random'] & random_data['agent_random']]
    csv_data = csv_data[(csv_data['agent'] == "first_0") & ~csv_data['vs_random'] & ~csv_data['agent_random']]
    #print(data)
    all_envs = sorted(set(csv_data['experiment']))
    print(all_envs)

    # all_env_names = {env: get_env_name(env) for env in all_envs}
    # all_envs = sorted(all_envs, key=str.lower)

    plot_ind = 1
    for env in all_envs:
        print("plotted")
        df = csv_data[(csv_data['experiment'] == env)]
        plt.subplot(8,3,plot_ind)
        rand_reward = random_data[(random_data['game'] == get_env_name(env))].iloc[0]['agent1_rew']
        print(rand_reward)
        #df = pd.read_csv(os.path.join(data_path, env+'.csv'))
        df = df[['checkpoint', "agent1_rew"]]
        data = df.to_numpy()
        #filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/110)+2, 5)
        filtered = data[:,1]
        line, = plt.plot(data[:, 0], filtered, label=env, linewidth=0.6, color='#0530ad', linestyle='-')
        rand_line, = plt.plot(data[:, 0],rand_reward*np.ones_like(data[:, 0]), label=env, linewidth=0.6, color='#A0522D', linestyle='-')
        plt.xlabel('Steps', labelpad=1)
        plt.ylabel('Average Total Reward', labelpad=1)
        plt.title(get_exp_label(env))
        #plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
        #plt.xlim(0, 60000)
        #plt.yticks(ticks=[0,150,300,450,600],labels=['0','150','300','450','600'])
        #plt.ylim(-150, 750)
        plt.tight_layout()
        #plt.legend(loc='lower center', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.5, -0.6))
        plt.margins(x=0)
        plot_ind += 1


    plt.figlegend([line,rand_line],['Trained Agent vs Random Agent', "Random Agent vs Random Agent"], fontsize='x-large', loc='lower center', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.68,0.06))
    plt.savefig("atari_results.pgf", bbox_inches = 'tight',pad_inches = .025)
    plt.savefig("atari_results.png", bbox_inches = 'tight',pad_inches = .025, dpi=600)

if __name__ == "__main__":
    main()
