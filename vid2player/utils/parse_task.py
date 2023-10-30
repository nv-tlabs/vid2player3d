# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from env.tasks.physics_mvae_controller import PhysicsMVAEController
from env.tasks.physics_mvae_controller_dual import PhysicsMVAEControllerDual
from env.tasks.mvae_controller_vis import MVAEControllerVis
from env.tasks.mvae_controller_vis_dual import MVAEControllerVisDual
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper

import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [MVAERecover]")

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        task = eval(cfg['name'])(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train['params']['config'].get("clip_observations", np.inf), cfg_train['params']['config'].get("clip_actions_val", np.inf))

    return task, env
