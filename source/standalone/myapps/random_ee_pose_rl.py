# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrate a single-arm manipulator.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/my_demo.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a single-arm manipulator.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import random
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from common import RoboticIkRlEnvCfg


def main():
    """Main function."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # # parse the arguments
    # env_cfg = RoboticIkRlEnvCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    # controller 
    total_action_dim = env.action_manager.total_action_dim
    print(f"Total action dim: {total_action_dim}")
    # robot 
    robot = env.scene["robot"]

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            # sample in position, and quaternion space e.g. (x, y, z, qx, qy, qz, qw)
            # Define the ranges for x, y, and z
            x_range = [0.4, 0.5]
            y_range = [-0.5, 0.5]
            z_range = [0, 0.7]

            # Sample the coordinates from the specified ranges
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            z = random.uniform(z_range[0], z_range[1])

            # Set the quaternions to [0.0, 1.0, 0.0, 0.0]
            quaternion = [0.0, 1.0, 0.0, 0.0]
            # Combine the position and orientation into a single pose
            end_effector_pose = [x, y, z] + quaternion
            end_effector_pose_tensor = torch.tensor(end_effector_pose, device=robot.device)
            # pick a random goal from the list
            ik_commands = torch.zeros(env_cfg.scene.num_envs, total_action_dim, device=robot.device)
            # create a torch tensor from ee_goals
            ik_commands[:] = end_effector_pose_tensor
            # step the environment
            obs, rew, terminated, truncated, info_ = env.step(ik_commands)
            # print current orientation of pole
            print("[Env 0]: joints: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
