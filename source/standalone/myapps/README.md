# README Isaac Lab robotic ultrasound

# Requirements
1. Follow [instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install IsaacLab. [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is suggested for virtual environment setup.


2. Download [Omniverse Launcher](https://developer.nvidia.com/omniverse#section-getting-started)

3. Create a [local Nucleus server ](https://docs.omniverse.nvidia.com/nucleus/latest/workstation/installation.html#installing-nucleus-workstation)

4. [Download](https://drive.google.com/drive/folders/1txXR1aXtIVWgvdAy_q21SLqIPs_3hs0j) the customized USD assets

5. Extract the Usd files and move the ultrasound folder into your local nucleus folder, such that the files are available at `omniverse://localhost/Library/ultrasound/`

6. Double-check the filepaths to the custom USDs, e.g. franka_realsense_no_world.usd and organ_rigid.usda





## Instructions 
Before launching any of the apps, activate the environment
```sh
cd "C:\code\forks\IsaacLab"
conda activate isaaclab
```
## Apps

### organ environment
environment to load organs and robotic manipulator
```sh
```

### State Machine
Without any learning, we can control the robot to reach positions, or full poses. These poses can depend on the body pose. This script reaches a position above the torso, with the camera pointed down, looking at the torso.

```sh
python "source/standalone/myapps/reach_torso.py" --task Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0 --enable_cameras
```
You can launch multiple instances (2) by appending: 
`--num_envs 2`

Todo: extent into a state machine, with multiple goals




### Training

The example from the tutorial
```sh
python source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64

# Franka draw 
python source/standalone/workflows/rl_games/train.py --task Isaac-Open-Drawer-Franka-v0 --num_envs 2
```

Ours: 
- Increase `--num_envs 2` as needed
- save videos `--video`
```sh
python source/standalone/myapps/RLTraining.py --task Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0 --enable_cameras --num_envs 64 --video
```

See logs: 
```sh
python -m tensorboard.main --logdir logs/sb3/Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0/2024-12-12_14-40-21/
```

### Play

```sh
python source/standalone/myapps/play.py --task Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0 --num_envs 32 --use_last_checkpoint --enable_cameras 
```

optionally define the path to the checkpoint to load. 
```sh
python source/standalone/myapps/play.py --task Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0 --num_envs 32 --checkpoint "/home/maxofir/repos/forks/IsaacLab/logs/sb3/Isaac-Robotic-Ultrasound-Franka-IK-RL-Abs-v0/2024-12-16_14-48-00/model_155000_steps.zip" --enable_cameras 
```