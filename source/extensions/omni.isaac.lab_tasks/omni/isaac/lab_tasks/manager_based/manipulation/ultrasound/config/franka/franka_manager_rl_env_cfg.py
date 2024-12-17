# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import CameraCfg

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)

from omni.isaac.lab_assets import FRANKA_PANDA_REALSENSE_CFG, FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

from ... import mdp


@configclass
class RoboticSoftCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # body
    # spawn the organ model onto the table, it needs to be scaled (1/10 of an inch?)
    # the model with _rigid was modified in USDComposer to have rigid body properties.
    # Leaving the props empty will use the default values.
    organs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.4, -0.1]),
        spawn=sim_utils.UsdFileCfg(
            # usd_path="omniverse://localhost/Library/test/test_cube.usd",
            usd_path="omniverse://localhost/Library/ultrasound/phantom/skin_tone_rigid.usd",
            scale=(0.00254, 0.00254, 0.00254),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # articulation
    # robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Robot"
    # )
    # alternative robot without camera
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link7/front_cam",
        update_period=0.0,
        history_length=1,  # Must be > 0
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # corresponds to 180 degree rotation around x-axis
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, -0.5), rot=(1, 0, 0, 0), convention="ros"
        ),
    )

    # Frame definitions for the goal frame
    goal_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs/models_topo_blender",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/goal_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/organs/models_topo_blender",
                name="goal_frame",
                offset=OffsetCfg(
                    pos=(0.0, -0.25, 1.0),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )




##
# MDP settings
##
@configclass
class EmptyCommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

# We can later use this to alternate goals for the robot
# It's not strictly necessary. The agent can learn based on observations, actions and rewards.
    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.45),
            pos_y=(0.0, 0.0),
            pos_z=(0.75, 0.75),
            roll=(1.5708, 1.5708),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # set the joint positions as target
    # joint_pos_des = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)
    # overwrite in post_init
    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    # todo: add camera as observation term

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # Add camera observation
        # camera_rgbd = ObsTerm(func=mdp.camera_rgbd_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # the reset scene to event function already resets all rigid objects and articulations to rheir default states.
    # this needs to be executed before any other reset function, to not overwrite the reset scene to default.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # the second reset only affects the organ body, and adds a random offset to the organ body, w.r.t to the current position.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0, -0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("organs"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    reaching_object = RewTerm(func=mdp.object_ee_distance, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # distance_to_patient = RewTerm(func=mdp.distance_to_patient, weight=1.0)
    # align_ee_patient = RewTerm(func=mdp.align_ee_patient, weight=1.0)

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    robot_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "bounds": (-3.0, 3.0),
        },
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class RoboticEnvIkCfg(ManagerBasedEnvCfg):
    """Configuration for the robotic ultrasound environment."""

    # scene settings
    scene: RoboticSoftCfg = RoboticSoftCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    commands = EmptyCommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
        self.episode_length_s = 5.0

        # configure the action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )
        # self.commands.target_pose.body_name = "panda_hand"

        # set a different start pose for the robot
        joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.01,
            "panda_joint3": 0.0,
            "panda_joint4": -1.0,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }
        self.scene.robot.init_state.joint_pos = joint_pos

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class RoboticEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the robotic ultrasound environment."""

    # scene settings
    scene: RoboticSoftCfg = RoboticSoftCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )


@configclass
class RoboticIkRlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the robotic ultrasound environment."""

    # Scene settings
    scene: RoboticSoftCfg = RoboticSoftCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # The command generator should ...
    commands: CommandsCfg = EmptyCommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.episode_length_s = 5

        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation

        # # set a different start pose for the robot
        # joint_pos={
        #     "panda_joint1": 0.0,
        #     "panda_joint2": -0.01,
        #     "panda_joint3": 0.0,
        #     "panda_joint4": -1.0,
        #     "panda_joint5": 0.0,
        #     "panda_joint6": 3.037,
        #     "panda_joint7": 0.741,
        #     "panda_finger_joint.*": 0.04,
        # }
        # self.scene.robot.init_state.joint_pos = joint_pos

        # configure the action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )

        # Set the body name for the end effector
        # self.commands.target_pose.body_name = "panda_hand"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/ee_frame"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
