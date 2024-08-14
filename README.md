# Learning Physically Simulated Tennis Skills from Broadcast Videos

<strong>Haotian zhang</strong>, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, Kayvon Fatahalian

SIGGRAPH 2023 (best paper honorable mention) 

[Paper](https://research.nvidia.com/labs/toronto-ai/vid2player3d/data/tennis_skills_main.pdf) |
[Project](https://research.nvidia.com/labs/toronto-ai/vid2player3d/) |
[Video](https://youtu.be/ZZVKrNs7_mk) 

<img src="doc/teaser.png"/>

### Note: the current release provides the implementation of the hierarchical controller, including the low-level imitation policy, motion embedding and the high-level planning policy, as well as the environment setup in IsaacGym. Unfortunately, the demo can NOT run because the trained models are currently not available due to license issues. 

# News
[2023/11/28] Training code for the low-level policy is released.

[2023/11/01] Demo code for the hierarchical controller is released.

# Environment setup

### 1. Download IsaacGym and create python virtual env
You can download IsaacGym Preview Release 4 from the official [site](https://developer.nvidia.com/isaac-gym).
Then download Miniconda3 from [here](https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh).
Create a conda virtual env named `rlgpu` by running `create_conda_env_rlgpu.sh` from IsaacGym, either python3.7 or python3.8 works.
Note you might need to run the following command or add it to your `.bashrc` if you encounter the error `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` when running IsaacGym.
```
export LD_LIBRARY_PATH=<YOUR CONDA PATH>envs/rlgpu/lib/
``` 

### 2. Install dependencies 
Enter the created virtual env and run the install script.
```
conda activate rlgpu
bash install.sh
```
To install additional dependencies for the low-level policy, follow the instructions in [install_embodied_pose.sh](install_embodied_pose.sh). 

### 3. Install [smpl_visualizer](https://github.com/Haotianz94/smpl_visualizer) for visualizing results
Git clone and then run 
```
bash install.sh
```

### 4. Download data/checkpoints

Download [data](https://drive.google.com/drive/folders/1zuDOJWtjjGOL5ZWPlErv39ORfg0NEi_s?usp=sharing) into `vid2player3d/data`.

Download checkpoints of motion embedding and trained polices into `vid2player3d/results` (currently unavailable).

Download SMPL by first registering [here](https://smpl.is.tue.mpg.de/login.php) and then download the [models](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip) (male and female models) into `smpl_visualizer/data/smpl` and rename the files as `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`.

For training the low-level policy, also copy the smpl model files into `vid2player3d/data/smpl`.

# Demo
These demos require trained models which are currently unavailable.
### Single player
In the single player setting, the player will react to consecutive incoming tennis balls from the other side.
The script below runs the simulation and renders the result online. The simulation will be reset after 300 frames. You can change the player by chaning`--cfg` to `djokovic` or `nadal`. 
```
python vid2player/run.py --cfg federer --rl_device cuda:0 --test --num_envs 1 --episode_length 300 --seed 0 --checkpoint latest --enable_shadow
```

The script below will run the simulations in batch and render the result videos offline and saved into `out/video`. You can also change `--record` to `--record_scenepic`, which will save the result into an interactive html file under `out/html`. Note that the saved html file is large and may take seconds to load.
```
python vid2player/run.py --cfg federer --rl_device cuda:0 --test --num_envs 8192 --episode_length 300 --seed 0 --checkpoint latest --select_best --enable_shadow --num_rec_frames 300 --num_eg 5 --record --headless
```

### Dual player
In the dual player setting, the two players will play tennis rally against each other.
The script below runs the simulation and renders the result online. The simulation will be reset if the ball is missed or out. You can change the players by changing `--cfg` to `nadal_djokovic`. More player settings will be added soon. 
```
python vid2player/run.py --cfg federer_djokovic --rl_device cuda:0 --test --num_envs 2 --episode_length 10000 --seed 0 --checkpoint latest --enable_shadow
```

The script below will run the simulations in batch and render the result videos offline and saved into `out/video`.
```
python vid2player/run.py --cfg federer_djokovic --rl_device cuda:0 --test --num_envs 8192 --episode_length 10000 --seed 0 --checkpoint latest --enable_shadow --headless --num_rec_frames 600 --num_eg 5 --record
```

# Training

### Low-level policy
We provide the code for training the low-level policy in [embodied_pose](embodied_pose). As described in the paper, the low-level policy is trained in two stages using AMASS motions and tennis motions. You can run the following script to execute the two-stage training (assuming the motion data are available).
```
python embodied_pose/run.py --cfg amass_im --rl_device cuda:0 --headless
python embodied_pose/run.py --cfg djokovic_im --rl_device cuda:0 --headless
```
[convert_amass_isaac.py](uhc/utils/convert_amass_isaac.py) shows how to convert the AMASS motion dataset into the format that can be used for our training code.

### Motion embedding
We provide code for training the motion embedding in [vid2player/motion_vae](vid2player/motion_vae/) (assuming the motion data is organized in the format described in [Video3DPoseDataset](vid2player/motion_vae/dataset.py)).

### High-level policy
We also provide code for training the high-level policy in [vid2player](vid2player). As described in the paper, we design a curriculum trained in three stages. You can run the following script to execute the curriculum training (assuming the checkpoints for the low-leve policy and motion embedding are available).
```
python vid2player/run.py --cfg federer_train_stage_1 --rl_device cuda:0 --headless
python vid2player/run.py --cfg federer_train_stage_2 --rl_device cuda:0 --headless
python vid2player/run.py --cfg federer_train_stage_3 --rl_device cuda:0 --headless
```


# Citation
```
@article{
  zhang2023vid2player3d,
  author = {Zhang, Haotian and Yuan, Ye and Makoviychuk, Viktor and Guo, Yunrong and Fidler, Sanja and Peng, Xue Bin and Fatahalian, Kayvon},
  title = {Learning Physically Simulated Tennis Skills from Broadcast Videos},
  journal = {ACM Trans. Graph.},
  issue_date = {August 2023},
  numpages = {14},
  doi = {10.1145/3592408},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {physics-based character animation, imitation learning, reinforcement learning},
}
```

# References
This repository is built on top of the following repositories:
* Low-level imitation policy is adapted from [EmbodiedPose](https://github.com/ZhengyiLuo/EmbodiedPose)
* Motion embedding is adapted from [character-motion-vaes](https://github.com/electronicarts/character-motion-vaes)
* RL environment in IsaacGym is adapted from [ASE](https://github.com/nv-tlabs/ASE/)
  
Here are additional references for reproducing the video annotation pipeline:
* Player detection and tracking: [Yolo4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
* 2D Pose keypoint detection: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
* 3D Pose estimation and mesh recovery: [HybrIK](https://github.com/Jeff-sjtu/HybrIK)
* 2D foot contact [detection](https://github.com/yul85/movingcam)
* Global root trajectory optimization: [GLAMR](https://github.com/NVlabs/GLAMR)
* Tennis court line [detection](https://github.com/gchlebus/tennis-court-detection)


# Contact
For any question regarding this project, please contact Haotian Zhang via haotianz@nvidia.com.
