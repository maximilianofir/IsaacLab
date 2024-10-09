# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni
from omni.isaac.lab_assets import (
    FRANKA_PANDA_CFG)
from omni.isaac.lab.assets import Articulation




def main():
    """Main function."""
    usd_path = "omniverse://localhost/Library/OR_scene_imagesTr_liver_27_relabel_resample1_syn_seed6_postprocess/main_scene.usd"
    omni.usd.get_context().open_stage(usd_path)
    print("Loading stage...")
    from omni.isaac.core.utils.stage import is_stage_loading

    while is_stage_loading():
        simulation_app.update()
    print("Loading Complete")
    
    # Load the Franka robot and the target object
    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_panda = Articulation(cfg=franka_arm_cfg)

    
    omni.timeline.get_timeline_interface().play()
    while simulation_app.is_running():
        simulation_app.update()

    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()

    # # Initialize the simulation context
    # sim_cfg = SimulationCfg(dt=0.01)
    # sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # # Play the simulator
    # sim.reset()
    # # Now we are ready!
    # print("[INFO]: Setup complete...")

    # # Simulate physics
    # while simulation_app.is_running():
    #     # perform step
    #     sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
