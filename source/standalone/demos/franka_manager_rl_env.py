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

from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab.envs.mdp as mdp

from omni.isaac.lab.managers import SceneEntityCfg


##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import FRANKA_PANDA_CFG

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
    # articulation
    # -- Robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube_deform : DeformableObjectCfg = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_deform",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.4, 0.4, 0.4),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 1.05)),
            debug_vis=True,
        )

    # # add cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 2.05)),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5000.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(),

        ),
    )

    # # Set Cube as object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 1.055], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="cube",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
    
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # generate random joint positions
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
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



# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     """Runs the simulation loop."""
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     # cube_object = scene["cube"]
#     robot = scene["franka_arm"]

#     # Nodal kinematic targets of the deformable bodies
#     # nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()
#     # Simulate physics
#     while simulation_app.is_running():
#         # reset
#         if count % 200 == 0:
#             # reset counters
#             sim_time = 0.0
#             count = 0
#             # reset the scene entities
#             # root state
#             root_state = robot.data.default_root_state.clone()
#             robot.write_root_state_to_sim(root_state)
#             # set joint positions
#             joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#             robot.write_joint_state_to_sim(joint_pos, joint_vel)
#             # clear internal buffers
#             robot.reset()
#             scene.reset()
#             # # reset root state
#             # root_state = object.data.default_root_state.clone()
#             # # sample a random position on a cylinder around the origins
#             # root_state[:, :3] += origins
#             # root_state[:, :3] += math_utils.sample_cylinder(
#             #     radius=0.1, h_range=(0.25, 0.5), size=object.num_instances, device=object.device
#             # )
#             # # write root state to simulation
#             # object.write_root_state_to_sim(root_state)
#             # # reset buffers
#             # object.reset()
#             # print("[INFO]: Resetting robots state...")

#             # # reset the nodal state of the object
#             # nodal_state = cube_object.data.default_nodal_state_w.clone()
#             # # apply random pose to the object
#             # pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
#             # quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
#             # nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

#             # # write nodal state to simulation
#             # cube_object.write_nodal_state_to_sim(nodal_state)

#             # # write kinematic target to nodal state and free all vertices
#             # nodal_kinematic_target[..., :3] = nodal_state[..., :3]
#             # nodal_kinematic_target[..., 3] = 1.0
#             # cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

#             # # reset buffers
#             # cube_object.reset()
#         # apply random actions to the robots
#         # generate random joint positions
#         joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
#         joint_pos_target = joint_pos_target.clamp_(
#             robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
#         )
#         # apply action to the robot
#         robot.set_joint_position_target(joint_pos_target)
#         # write data to sim
#         # robot.write_data_to_sim()
#         # -- write data to sim

#         scene.write_data_to_sim()

#         # # update the kinematic target for cubes at index 0 and 3
#         # # we slightly move the cube in the z-direction by picking the vertex at index 0
#         # nodal_kinematic_target[[0], 0, 2] += 0.001
#         # # set vertex at index 0 to be kinematically constrained
#         # # 0: constrained, 1: free
#         # nodal_kinematic_target[[0], 0, 3] = 0.0
#         # # write kinematic target to simulation
#         # cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
#         # # write internal data to simulation
#         # cube_object.write_data_to_sim()

#         # perform step
#         sim.step()
#         # update sim-time
#         sim_time += sim_dt
#         count += 1
#         # update buffers
#         robot.update(sim_dt)
#         # cube_object.update(sim_dt)


def main():
    """Main function."""
    env_cfg = RoboticEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    # setup base environment

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counters
            count += 1
    # env = ManagerBasedEnv(cfg=env_cfg)
    # # Initialize the simulation context
    # sim_cfg = sim_utils.SimulationCfg()
    # sim = sim_utils.SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # # design scene
    # scene_cfg = RoboticSoftCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # scene = InteractiveScene(scene_cfg)
    # sim.reset()
    # # Now we are ready!
    # print("[INFO]: Setup complete...")
    # # Run the simulator

    # run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
