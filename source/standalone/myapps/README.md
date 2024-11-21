# README Isaac Lab robotic ultrasound
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

```sh
python "source\standalone\myapps\random_ee_pose_rl.py" --task Isaac-Robotic-Ultrasound-Franka-IK-Abs-v0
```