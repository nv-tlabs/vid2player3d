# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from env.tasks.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

class VecTaskCPUWrapper(VecTaskCPU):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations, clip_actions)
        return

class VecTaskGPUWrapper(VecTaskGPU):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        return


class VecTaskPythonWrapper(VecTaskPython):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
