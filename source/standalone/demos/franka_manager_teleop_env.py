# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrate a single-arm manipulator.

.. code-block:: bash

    # Usage
    .\isaaclab.bat -p source\standalone\demos\franka_manager_teleop_env.py --enable_camerass 
    --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

#This is a procedure to launch the ISAAC SIM via python CODE 

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg, AssetBaseCfg, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm

#from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
#import omni.isaac.lab.envs.mdp as mdp

from omni.isaac.lab.managers import SceneEntityCfg

#teleoperation includes 
import gymnasium as gym
import omni.log
from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.ultrasound import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

# isort: on

# Table_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"))

#from franka_manager_env import RoboticSoftCfg
            

def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Rel" in args_cli.task:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        test = gripper_vel
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        return torch.concat([delta_pose, gripper_vel], dim=1)
        # compute actions
    if "Abs" in args_cli.task:
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        test = gripper_vel
        gripper_vel[:] = 1
        return torch.concat([delta_pose, gripper_vel, test], dim=1)
    else:
         return torch.concat([delta_pose, gripper_vel], dim=1)
    

def main():
    
    """Main function."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.sim.device = args_cli.device

    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Ultrasound" in args_cli.task:
        print('Test')
    #     # set the resampling time range to large number to avoid resampling
        #env_cfg.target_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.robot_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "bounds": (-3.0, 3.0),
        },)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            env.step(actions)
            print('I am here')     

    print('I am here2')     
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
