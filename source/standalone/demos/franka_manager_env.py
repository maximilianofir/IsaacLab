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
    # articulation
    # -- Robot
    # robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link7/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # corresponds to 180 degree rotation around x-axis
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, -0.5), rot=(1, 0, 0, 0), convention="ros"),
    )

    # initial position from teddy_bear example
    cube_deform : DeformableObjectCfg = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_deform",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.4, 0.4, 0.4),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05)),
            debug_vis=True,
        )

    # # add cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
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


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # generate random joint positions
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

    # set the joint positions as target
    joint_pos_des = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


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






def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    deformable_object = scene["cube_deform"]
    rigid_object = scene["cube"]
    # env_cfg = RoboticEnvCfg()
    # env = ManagerBasedEnv(cfg=env_cfg)

    # # setup base environment
    # robot = env.scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    # add a marker for the sensor 
    sensor_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/sensor"))

    # Define goals for the arm
    ee_goals = [
        [0.1, 0.1, 1.0, 0.707, 0, 0.707, 0],
        [0.2, 0.1, 1.0, 0.707, 0.707, 0.0, 0.0],
        [0.3, 0.1, 1.0, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    print(f"envs: {scene.num_envs}", f"diff_ik_controller.action_dim: {diff_ik_controller.action_dim}")

    ik_commands[:] = ee_goals[current_goal_idx]
    
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand", "panda_link7"], preserve_order=True)
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    print(f"robot_entity_cfg.body_ids: {robot_entity_cfg.body_ids}")
    print(f"robot_entity_cfg.joint_ids: {robot_entity_cfg.joint_ids}")
    print(f"robot_entity_cfg: {robot_entity_cfg}")

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Nodal kinematic targets of the deformable bodies
    deformable_object_kinematic_target = deformable_object.data.nodal_kinematic_target.clone()

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actionss
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # print joint positions and joint ids
            print("[INFO]: Joint positions: ", joint_pos)
            print("[INFO]: Joint ids: ", robot_entity_cfg.joint_ids)
            print("[INFO]: Joint positions des: ", joint_pos_des)
            print("-" * 80)
            print("[INFO]: Resetting environment...")
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            
            # reset the objects
            # reset the nodal state of the object
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            # apply random pose to the object
            pos_w = torch.rand(deformable_object.num_instances, 3, device=sim.device) * 0.1 
            quat_w = math_utils.random_orientation(deformable_object.num_instances, device=sim.device)
            nodal_state[..., :3] = deformable_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write nodal state to simulation
            deformable_object.write_nodal_state_to_sim(nodal_state)
            
            # reset the rigid object
            # reset root state
            root_state = rigid_object.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=rigid_object.num_instances, device=rigid_object.device
            )
            # write root state to simulation
            rigid_object.write_root_state_to_sim(root_state)
            # reset buffers
            rigid_object.reset()

            # write kinematic target to nodal state and free all vertices
            deformable_object_kinematic_target[..., :3] = nodal_state[..., :3]
            deformable_object_kinematic_target[..., 3] = 1.0
            deformable_object.write_nodal_kinematic_target_to_sim(deformable_object_kinematic_target)

            # reset buffers
            deformable_object.reset()
            rigid_object.reset()
            # reset scene
            scene.reset()
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            # print joint positions and joint ids
            print("[INFO SIM]: Joint positions: ", joint_pos)
            print("[INFO SIM]: Joint ids: ", robot_entity_cfg.joint_ids)
            print("[INFO SIM]: Joint positions des: ", joint_pos_des)
            
        # sample random actions
        # joint_pos_des = torch.randn_like(env.action_manager.action)
        # step the environment
        # obs, _ = env.step(joint_pos_des)
        # # print current orientation of pole
        # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
        # # update counters
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        camera_pose = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
        print(robot_entity_cfg.body_ids)
        
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        # update sensor marker
        sensor_marker.visualize(camera_pose[:, 0:3], camera_pose[:, 3:7])
        
        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        
# env = ManagerBasedEnv(cfg=env_cfg)
# # Initialize the simulation context


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = RoboticSoftCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
