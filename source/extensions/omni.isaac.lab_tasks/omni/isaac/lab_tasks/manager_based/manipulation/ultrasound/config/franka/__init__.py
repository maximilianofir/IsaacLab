# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from ..teleop import ik_rel_env_cfg
from . import franka_manager_rl_env_cfg
from . import agents
##
# Register Gym environments.
##

##
# Joint Position Control
##


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": franka_manager_rl_env_cfg.RoboticIkRlEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",

    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": franka_manager_rl_env_cfg.RoboticEnvIkCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Robotic-Ultrasound-Franka-Teleop-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaUltrasoundTeleopEnv,
#       "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

#gym.register(
#    id="Isaac-Robotic-Ultrasound-Franka-Teleop-IK-Rel-Play-v0",
#    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#    kwargs={
#        "env_cfg_entry_point": ik_rel_env_cfg.FrankaUltrasoundTeleopEnv_PLAY,
#        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#
#    },
#    disable_env_checker=True,
#)