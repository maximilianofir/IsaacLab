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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from dataclasses import MISSING

import numpy as np
import random
import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg, AssetBaseCfg, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm

from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab.envs.mdp as mdp

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import subtract_frame_transforms




##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_REALSENSE_CFG

# isort: on

# Table_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"))

@configclass
class RoboticSoftCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    
    # body
    # spawn the organ model onto the table, it needs to be scaled (1/10 of an inch?)
    organs = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Organs",
                          init_state=AssetBaseCfg.InitialStateCfg(pos=[0.2, 0.4, -0.1]),
                          spawn=sim_utils.UsdFileCfg(usd_path=R"C:\Users\mmoller\OneDrive - NVIDIA Corporation\Documents\projects\ImFusion\shared\roboticUltrasound\ultrasound\environment\organ.usda",
                                                     scale=(0.00254, 0.00254, 0.00254)))

    # articulation
    # -- Robot
    # robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # set the joint positions as target
    # joint_pos_des = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)
    # overwrite in post_init
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RoboticEnvCfg(ManagerBasedEnvCfg):
    # scene settings
    scene: RoboticSoftCfg = RoboticSoftCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
        
        # configure the action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        # self.actions.arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, use_default_offset=True)
        


def main():
    """Main function."""
    # parse the arguments
    env_cfg = RoboticEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)
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
            obs, _ = env.step(ik_commands)
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
