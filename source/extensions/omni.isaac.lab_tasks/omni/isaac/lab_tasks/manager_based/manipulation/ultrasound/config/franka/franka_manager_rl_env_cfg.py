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

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)

from omni.isaac.lab_assets import FRANKA_PANDA_REALSENSE_CFG

from omni.isaac.lab_tasks.manager_based.manipulation.ultrasound import mdp

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
            usd_path="omniverse://localhost/Library/ultrasound/environment/organ_rigid.usda",
            scale=(0.00254, 0.00254, 0.00254),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # articulation
    robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
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
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

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

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # distance_to_patient = RewTerm(func=mdp.distance_to_patient, weight=1.0)
    # align_ee_patient = RewTerm(func=mdp.align_ee_patient, weight=1.0)


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
    # No command generator
    commands: CommandsCfg = CommandsCfg()

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
        self.commands.target_pose.body_name = "panda_hand"

