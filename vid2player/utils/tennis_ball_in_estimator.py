from utils.tennis_ball import TennisBallGeneratorIsaac, simulate 

import numpy as np
from math import pi
import torch
import torch.nn.functional as F
from tqdm import tqdm


class traj_in_params:
    VEL_X_RANGE = (25, 30, 0.1)
    VEL_Y_RANGE = (5, 8, 0.1)
    VSPIN_RANGE = (5, 10, 0.1)
    HEIGHT_RANGE = (0.5, 2, 0.1)    

class TennisBallInEstimator():
    def __init__(self, ball_traj_file):
        self._ball_traj = np.load(ball_traj_file)
        self._ball_traj = torch.from_numpy(self._ball_traj)
        self.params = traj_in_params

    def get_ball_traj_index(self, height, vel_x, vel_y, vspin):
        VEL_X_RANGE, VEL_Y_RANGE, VSPIN_RANGE, HEIGHT_RANGE = \
            self.params.VEL_X_RANGE, self.params.VEL_Y_RANGE, self.params.VSPIN_RANGE, self.params.HEIGHT_RANGE
        
        height = torch.clamp(height, HEIGHT_RANGE[0], HEIGHT_RANGE[1]-HEIGHT_RANGE[2])
        vel_x = torch.clamp(vel_x, VEL_X_RANGE[0], VEL_X_RANGE[1]-VEL_X_RANGE[2])
        vel_y = torch.clamp(vel_y, VEL_Y_RANGE[0], VEL_Y_RANGE[1]-VEL_Y_RANGE[2])
        vspin = torch.clamp(vspin, VSPIN_RANGE[0], VSPIN_RANGE[1]-VSPIN_RANGE[2])
        dim = (
            (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) / HEIGHT_RANGE[2],
            (VEL_X_RANGE[1] - VEL_X_RANGE[0]) / VEL_X_RANGE[2],
            (VEL_Y_RANGE[1] - VEL_Y_RANGE[0]) / VEL_Y_RANGE[2],
            (VSPIN_RANGE[1] - VSPIN_RANGE[0]) / VSPIN_RANGE[2],
        )
        index = torch.round((height - HEIGHT_RANGE[0]) / HEIGHT_RANGE[2]) * dim[1] * dim[2] * dim[3] \
            + torch.round((vel_x - VEL_X_RANGE[0]) / VEL_X_RANGE[2]) * dim[2] * dim[3] \
            + torch.round((vel_y - VEL_Y_RANGE[0]) / VEL_Y_RANGE[2]) * dim[3] \
            + torch.round((vspin - VSPIN_RANGE[0]) / VSPIN_RANGE[2])
        
        height = torch.round((height - HEIGHT_RANGE[0]) / HEIGHT_RANGE[2]) * HEIGHT_RANGE[2] + HEIGHT_RANGE[0]
        vel_x = torch.round((vel_x - VEL_X_RANGE[0]) / VEL_X_RANGE[2]) * VEL_X_RANGE[2] + VEL_X_RANGE[0]
        vel_y = torch.round((vel_y - VEL_Y_RANGE[0]) / VEL_Y_RANGE[2]) * VEL_Y_RANGE[2] + VEL_Y_RANGE[0]
        vspin = torch.round((vspin - VSPIN_RANGE[0]) / VSPIN_RANGE[2]) * VSPIN_RANGE[2] + VSPIN_RANGE[0]
        
        return index.cpu().long(), (height, vel_x, vel_y, vspin)
    
    def estimate(self, ball_states, adjust=False):
        device = ball_states.get_device()

        height = ball_states[:, 2]
        vel_x = ball_states[:, 7:9].norm(dim=-1)
        dir = ball_states[:, 7:9] / vel_x.view(-1, 1)
        vel_y = ball_states[:, 9]
        vspin = ball_states[:, 10:13].norm(dim=1) / (pi * 2)
        
        traj_index, (height, vel_x, vel_y, vspin) = self.get_ball_traj_index(height, vel_x, vel_y, vspin)

        traj = self._ball_traj[traj_index].clone().to(device)
        # transform traj
        traj_trans = torch.cat([traj[:, :, :1] * dir.view(-1, 1, 2) + ball_states[:, :2].view(-1, 1, 2), traj[:, :, 1:]], dim=-1)
        traj_trans[:, :, :2] *= -1

        # update states
        ball_states_in = ball_states.clone()
        ball_states_in[:, :2] *= -1
        ball_states_in[:, 2] = height
        ball_states_in[:, 7:9] = - vel_x.view(-1, 1) * dir
        ball_states_in[:, 9] = vel_y
        ball_states_in[:, 10:13] = vspin.view(-1, 1) * pi * 2 * F.normalize(
        torch.cross(ball_states_in[:, 7:10], torch.FloatTensor([0, 0, -1]).repeat(ball_states.shape[0], 1).to(device)), dim=1)

        ball_states_out = ball_states_in.clone()
        ball_states_out[:, :2] *= -1
        ball_states_out[:, 7:9] *= -1
        ball_states_out[:, 10:13] = vspin.view(-1, 1) * pi * 2 * F.normalize(
        torch.cross(ball_states_out[:, 7:10], torch.FloatTensor([0, 0, -1]).repeat(ball_states.shape[0], 1).to(device)), dim=1)

        return traj_trans, ball_states_in, ball_states_out


def generate_incoming_trajectory(traj_path, params):
    VEL_X_RANGE, VEL_Y_RANGE, VSPIN_RANGE, HEIGHT_RANGE = \
        params.VEL_X_RANGE, params.VEL_Y_RANGE, params.VSPIN_RANGE, params.HEIGHT_RANGE

    ball_gen = TennisBallGeneratorIsaac({}, is_train=True, need_reset=False)
    device = torch.device('cuda:0') if ball_gen.args.use_gpu_pipeline else torch.device('cpu')

    height_range = np.arange(*HEIGHT_RANGE)
    vel_y_range = np.arange(*VEL_X_RANGE)
    vel_z_range = np.arange(*VEL_Y_RANGE)
    vspin_range = np.arange(*VSPIN_RANGE)
    batch_size_tuple = (len(height_range), len(vel_y_range), len(vel_z_range), len(vspin_range))
    batch_size = len(height_range) * len(vel_z_range) * len(vel_y_range) * len(vspin_range)
    
    batch_height = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    batch_vel_z = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    batch_vel_y = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    batch_vspin = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    for i, h in enumerate(height_range):
        batch_height[i, :, :, :] = h
    for i, vel_y in enumerate(vel_y_range):
        batch_vel_y[:, i, :, :] = vel_y
    for i, vel_z in enumerate(vel_z_range):
        batch_vel_z[:, :, i, :] = vel_z
    for i, vspin in enumerate(vspin_range):
        batch_vspin[:, :, :, i] = vspin
    batch_height = batch_height.view(-1)
    batch_vel_y = batch_vel_y.view(-1)
    batch_vel_z = batch_vel_z.view(-1)
    batch_vspin = batch_vspin.view(-1)
    
    num_ball = ball_gen.num_env
    launch_pos = torch.zeros((num_ball, 3), device=device, dtype=torch.float32)
    # HACK for speed up (extremely slow if all start from the same pos)
    launch_pos[:, 0] = torch.arange(num_ball) / 1000

    traj_all = None
    for num_iter in tqdm(range(batch_size // num_ball)):
        launch_pos[:, 2] = batch_height[num_iter*num_ball: (num_iter+1)*num_ball]

        launch_vel = torch.zeros((num_ball, 3), device=device, dtype=torch.float32)
        launch_vel[:, 1] = batch_vel_y[num_iter*num_ball: (num_iter+1)*num_ball]
        launch_vel[:, 2] = batch_vel_z[num_iter*num_ball: (num_iter+1)*num_ball]
        
        launch_vspin = batch_vspin[num_iter*num_ball: (num_iter+1)*num_ball]

        traj, bounce_pos, bounce_idx, pass_net = simulate(
            ball_gen.gym, ball_gen.sim, launch_pos, launch_vel, launch_vspin, 
            num_frames=50, device=device)

        # all trajs are straight out
        traj = traj[:, :, 1:]
        
        if traj_all is None:
            traj_all = traj
        else:
            traj_all = torch.cat([traj_all, traj], dim=0)

        np.save(traj_path, traj_all.cpu().numpy())


if __name__ == '__main__':

    generate_incoming_trajectory('data/ball_traj/ball_traj_in_dual.npy', params=traj_in_params)