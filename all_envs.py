from pettingzoo.atari import boxing_v1
from pettingzoo.atari import combat_tank_v1
from pettingzoo.atari import combat_plane_v1
from pettingzoo.atari import double_dunk_v2
from pettingzoo.atari import entombed_competitive_v2
from pettingzoo.atari import entombed_cooperative_v2
from pettingzoo.atari import flag_capture_v1
from pettingzoo.atari import joust_v2
from pettingzoo.atari import ice_hockey_v1
from pettingzoo.atari import maze_craze_v2
from pettingzoo.atari import mario_bros_v2
from pettingzoo.atari import othello_v2
from pettingzoo.atari import basketball_pong_v2
from pettingzoo.atari import pong_v2
from pettingzoo.atari import foozpong_v2
from pettingzoo.atari import quadrapong_v3
from pettingzoo.atari import volleyball_pong_v2
from pettingzoo.atari import space_invaders_v1
from pettingzoo.atari import space_war_v1
from pettingzoo.atari import surround_v1
from pettingzoo.atari import tennis_v2
from pettingzoo.atari import video_checkers_v3
from pettingzoo.atari import wizard_of_wor_v2
from pettingzoo.atari import warlords_v2

all_environments = {
    "boxing_v1": boxing_v1,
    "combat_tank_v1": combat_tank_v1,
    "combat_plane_v1": combat_plane_v1,
    "double_dunk_v2": double_dunk_v2,
    "entombed_cooperative_v2": entombed_cooperative_v2,
    "entombed_competitive_v2": entombed_competitive_v2,
    "flag_capture_v1": flag_capture_v1,
    "joust_v2": joust_v2,
    "ice_hockey_v1": ice_hockey_v1,
    "maze_craze_v2": maze_craze_v2,
    "mario_bros_v2": mario_bros_v2,
    "othello_v2": othello_v2,
    "pong_v2": pong_v2,
    "basketball_pong_v2": basketball_pong_v2,
    "foozpong_v2": foozpong_v2,
    "quadrapong_v3": quadrapong_v3,
    "volleyball_pong_v2": volleyball_pong_v2,
    "space_invaders_v1": space_invaders_v1,
    "space_war_v1": space_war_v1,
    "surround_v1": surround_v1,
    "tennis_v2": tennis_v2,
    "video_checkers_v3": video_checkers_v3,
    "wizard_of_wor_v2": wizard_of_wor_v2,
    "warlords_v2": warlords_v2,
}
builtin_envs = [
    "boxing_v1",
    "double_dunk_v2",
    "ice_hockey_v1",
    "pong_v2",
    "surround_v1",
    "tennis_v2",
]
# for e in all_environments.values():
#     e.env()
