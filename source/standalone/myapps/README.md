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

### franka manager environment
manager based environment with random motion on robot. The flag '--enable_cameras' allows for visualization of cameras in IsaacLab.

```sh
python "source\standalone\myapps\random_joint_pose.py"--enable_cameras
```

#### Cartesian control environment
Instead of randomly setting joint states, we can also use an inverse kinematics controller and sample target poses for the end-effector. 
This is done in  [franka_manager_ik_action_env.py](franka_manager_ik_action_env.py)
```sh
python python "source\standalone\myapps\random_ee_pose.py"
```

#### RL environment 
The EL environment is registered with gym. Therefore the API to call the model changes slightly. 

To register the environment, the configuration files were refactored and moved to a folder in source\extensions\omni.isaac.lab_tasks\omni\isaac\lab_tasks\manager_based\manipulation\ultrasound. 

The folder structure resembles that of similar applications in the same parent folder. 

The \_\_init\_\_.py at (source\extensions\omni.isaac.lab_tasks\omni\isaac\lab_tasks\manager_based\manipulation\ultrasound\config\franka\\\_\_init__.py) shows how to register the environments defined in  "source\extensions\omni.isaac.lab_tasks\omni\isaac\lab_tasks\manager_based\manipulation\ultrasound\config\franka\franka_manager_rl_env_cfg.py"

Windows:
```sh
python "source\standalone\myapps\random_ee_pose_rl.py" --task Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0
```
Linux:
```sh
python "source/standalone/myapps/random_ee_pose_rl.py" --task Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0
```

##### Training
The example from the tutorial
```sh
python source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64
```

Ours: 
```sh
python source/standalone/myapps/RLTraining.py --task Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0 --num_envs 1
```