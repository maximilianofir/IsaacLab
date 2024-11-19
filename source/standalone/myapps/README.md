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
python source\standalone\myapps\franka_manager_env.py --enable_cameras
```

#### Cartesian control environment
Instead of randomly setting joint states, we can also use an inverse kinematics controller and sample target poses for the end-effector. 
This is done in  [franka_manager_ik_action_env.py](franka_manager_ik_action_env.py)
```sh
python source\standalone\myapps\franka_manager_ik_action_env.py
```