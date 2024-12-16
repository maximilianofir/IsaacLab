# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrate a single-arm manipulator.

.. code-block:: bash

    # Usage

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

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers


def main():
    """Main function."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # reset environment at start
    env.reset()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # # parse the arguments
    # env_cfg = RoboticIkRlEnvCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    # env.get_wrapper_attr('action_manager')
    action_manager = env.get_wrapper_attr('action_manager')
    total_action_dim = action_manager.total_action_dim
    # print(f"Total action dim: {total_action_dim}")
    # robot 
    scene = env.get_wrapper_attr('scene')
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    # add a marker for the sensor 
    body_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/body"))


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

            # Observations
            # get the current pose of the target object
            object_data = env.unwrapped.scene["organs"].data
            # print("object_data:", object_data)
            object_position = object_data.root_pos_w
            object_orientation = object_data.root_quat_w
            # target position above the object position
            target_position = object_position + torch.tensor([0.0, -0.25, 1.0], device=robot.device) 

            # get the target position in each robot's base frame 
            target_position_robot_frame = target_position - env.unwrapped.scene.env_origins
            # # Get the desired position from the command manager
            # desired_position = env.unwrapped.command_manager.get_command("target_pose")[..., :3]
            # print("desired_position:", desired_position)
            quaternion = [0.0, 1.0, 0.0, 0.0]
            # repeat the quaternion for all the environments
            quaternion = torch.tensor(quaternion, device=robot.device).repeat(env_cfg.scene.num_envs, 1)
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

            # concatenate the position and orientation
            end_effector_pose_tensor = torch.cat([target_position_robot_frame, quaternion], dim=1) 
            # print("end_effector_pose:", end_effector_pose_tensor)
            # print("end_effector_pose.shape:", end_effector_pose_tensor.shape)
            # end_effector_pose_tensor = torch.tensor(end_effector_pose, device=robot.device)

            # pick a random goal from the list
            ik_commands = torch.zeros(env_cfg.scene.num_envs, total_action_dim, device=robot.device)
            # create a torch tensor from ee_goals
            ik_commands[:] = end_effector_pose_tensor
            # step the environment
            obs, rew, terminated, truncated, info_ = env.step(ik_commands)
            # print ("obs:", obs)
            # print current orientation of pole
            # print("[Env 0]: joints: ", obs["policy"][0][1].item())
            # update counter
            count += 1

            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(target_position[:, 0:3], object_orientation[:, :4])
            body_marker.visualize(object_position, object_orientation)
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
