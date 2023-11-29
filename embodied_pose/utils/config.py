# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import yaml
import glob
import numpy as np
import random

from isaacgym import gymapi
from isaacgym import gymutil

import torch


SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):

    cfg_path = f'embodied_pose/cfg/**/{args.cfg}.yaml'
    files = glob.glob(cfg_path, recursive=True)
    assert(len(files) == 1)
    cfg_file = files[0]

    with open(cfg_file, 'r') as f:
        cfg_all = yaml.load(f, Loader=yaml.SafeLoader)

    cfg = {'name': cfg_all['name'], 'test_name': cfg_all.get('test_name', 'HumanoidSMPLRefv1'), 'env': cfg_all['env'], 'sim': cfg_all['sim']}
    cfg_train = {'params': cfg_all['params']}

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    if args.task is not None:
        cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    cfg_dir = os.path.join('/tmp/embodied_pose' if args.tmp else args.results_dir, args.cfg)
    cfg['cfg_dir'] = cfg_dir

    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["config"]["load_checkpoint"] = True

    cfg_train["params"]["load_path"] = args.checkpoint
        
    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = cfg_train["params"]["config"]["args"] = args

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['network_path'] = network_path = os.path.join(cfg_dir, 'models') 
    cfg_train['params']['config']['log_path'] = log_path = os.path.join(cfg_dir, 'logs') 
    cfg_train['params']['config']['wandb_dir'] = cfg_dir
    cfg_train['params']['config']['train_dir'] = os.path.join(cfg_dir, 'train') 
    cfg_train['params']['config']['device'] = args.rl_device

    os.makedirs(network_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    if args.test:
        cfg['name'] = cfg['test_name']
    if args.motion_id is not None:
        cfg['env']['motion_id'] = args.motion_id
    if args.export_dataset is not None:
        cfg['env']['export_dataset'] = args.export_dataset
    
    cfg['env']['record'] = args.record
    if args.rec_fname is not None:
        cfg['env']['rec_fname'] = args.rec_fname
    if args.num_rec_frames is not None:
        cfg['env']['num_rec_frames'] = args.num_rec_frames
    if args.rec_fps is not None:
        cfg['env']['rec_fps'] = float(args.rec_fps)
    if args.camera is not None:
        cfg['env']['camera'] = args.camera

    if args.cpu_motion_lib:
        cfg['env']['gpu_motion_lib'] = not args.cpu_motion_lib

    if args.test_motion_file is not None:
        cfg['env']['test_motion_file'] = args.test_motion_file

    if args.motion_file_range is not None:
        new_range = [int(x) for x in args.motion_file_range.split('-')]
        if len(new_range) == 1:
            new_range.append(new_range[0] + 1)
        cfg['env']['motion_file_range'] = new_range

    if args.context_length is not None:
        cfg['env']['context_length'] = args.context_length

    return cfg, cfg_train


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "action": "store_true", "default": False,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--resume_id", "type": str, "default": None},
        {"name": "--checkpoint", "type": str, "default": None,
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": None,
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--results_dir", "type": str, "default": "results"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg", "type": str, "default": "Base", "help": "all-in-one configuration file (.yaml)"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--tmp", "action": "store_true", "default": False},
        {"name": "--no_log", "action": "store_true", "default": False},
        {"name": "--cpu_motion_lib", "action": "store_true", "default": False},
        {"name": "--test_motion_file", "type": str, "default": None},
        {"name": "--motion_id", "type": int, "default": None},
        {"name": "--motion_file_range", "type": str, "default": None},
        {"name": "--record", "action": "store_true", "default": False},
        {"name": "--num_rec_frames", "type": int, "default": None},
        {"name": "--rec_fps", "type": int, "default": None},
        {"name": "--rec_fname", "type": str, "default": None},
        {"name": "--rec_once", "action": "store_true", "default": False},
        {"name": "--vis_overlap", "action": "store_true", "default": False},
        {"name": "--vis_mode", "type": str, "default": 'normal'},
        {"name": "--context_length", "type": int, "default": None},
        {"name": "--camera", "type": str, "default": None},
        {"name": "--export_dataset", "type": str, "default": None},
    ]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    if args.resume_id is not None:
        args.resume = True
    args.sim_device = args.rl_device
    if args.use_gpu_pipeline:
        args.compute_device_id = int(args.rl_device[-1])
        args.graphics_device_id = int(args.rl_device[-1])
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
