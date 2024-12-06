# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import franka_manager_rl_env_cfg

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
    id="Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0", # The gym environment name.
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv", # the manager base class.
    kwargs={
        "env_cfg_entry_point": franka_manager_rl_env_cfg.RoboticIkRlEnvCfg, # the config class to be used to initialize the manager environment.
         #"sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##
