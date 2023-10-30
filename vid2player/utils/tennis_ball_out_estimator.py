from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
from math import pi
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.tennis_ball import * 


class traj_out_params:
    VEL_X_RANGE = (10, 65, 0.1)
    VEL_Y_RANGE = (-5, 10, 0.1)
    VSPIN_RANGE = (-10, 10, 0.2)
    TRAJ_X_RANGE = (0, 30, 0.5)
    TRAJ_Y_RANGE = (0, 3, 0.1)


def simulate_without_bounce(gym, sim, launch_pos, launch_vel, launch_vspin, params,
    control_freq_inv=2, num_frames=60, spin_scale=5, device=torch.device('cpu')):

    TRAJ_X_RANGE, TRAJ_Y_RANGE = params.TRAJ_X_RANGE, params.TRAJ_Y_RANGE

    def interpolate_x_batch(traj, t, x):
        x1, x2 = traj[ids_all, t-1, 0], traj[ids_all, t, 0]
        w = (x - x1) / (x2 - x1)
        y = traj[ids_all, t-1, 1] * (1-w) + traj[ids_all, t, 1] * w
        return y
    
    def interpolate_y_batch(traj, t, y):
        y1, y2 = traj[ids_all, t-1, 1], traj[ids_all, t, 1]
        w = (y - y1) / (y2 - y1)
        x = traj[ids_all, t-1, 0] * (1-w) + traj[ids_all, t, 0] * w
        tt = (t-1) * (1-w) + t * w
        return x, tt / (control_freq_inv * 30)

    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(actor_root_state)

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)

    num_ball = launch_pos.shape[0]
    num_env = root_state.shape[0]
    g_tensor = torch.FloatTensor([[0, 0, -1]]).repeat(num_env, 1).to(device)

    launch_ang_vel = launch_vspin.view(-1, 1) * pi * 2 * F.normalize(
        torch.cross(launch_vel, g_tensor[:num_ball]), dim=1)
    root_state[:, 0:3] = launch_pos
    root_state[:, 7:10] = launch_vel
    root_state[:, 10:13] = launch_ang_vel

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

    traj = None
    for t in range(num_frames + 1):
        for i in range(control_freq_inv):
            pos = root_state[:, 0:3]
            if traj is None:
                traj = pos.view(-1, 1, 3).clone()
            else:
                traj = torch.cat((traj, pos.view(-1, 1, 3)), dim=1)

            vel = root_state[:, 7:10]
            vel_scalar = vel.norm(dim=1).view(-1, 1)
            vel_norm = vel / vel_scalar
            vel_tan = torch.cross(vel_norm, g_tensor)
            vspin = root_state[:, 10:13].norm(dim=1) / (pi * 2)
            vspin = torch.where(launch_vspin > 0, vspin, vspin * -1).view(-1, 1)

            cd = get_cd(vel_scalar, vspin * spin_scale)
            cl = get_cl(vel_scalar, vspin * spin_scale)
            cl = cl * torch.where(vspin > 0, 
                -1 * torch.ones_like(cl),
                1 * torch.ones_like(cl)
            )

            force_drag = - kf * cd * vel_scalar * vel
            force_lift = - kf * cl * vel_scalar ** 2 * torch.cross(vel_tan, vel_norm) 
            
            forces = force_drag + force_lift
            forces[num_ball:, :] = 0
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), 
                None, gymapi.ENV_SPACE)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            
    traj_all = traj[:, :, 1:]
    traj_all[:, :, 1] -= traj_all[0, 0, 1]
    traj_x = torch.zeros((num_ball, int((TRAJ_X_RANGE[1] - TRAJ_X_RANGE[0]) / TRAJ_X_RANGE[2])), dtype=torch.float32, device=device)
    traj_y = torch.zeros((num_ball, int((TRAJ_Y_RANGE[1] - TRAJ_Y_RANGE[0]) / TRAJ_Y_RANGE[2]), 2), dtype=torch.float32, device=device)

    ids_all = torch.arange(num_ball)
    t = torch.zeros(num_ball, dtype=torch.int64, device=device)

    for x in torch.arange(*TRAJ_X_RANGE):
        while True:
            ids = (t < traj.shape[1] - 1) & (traj_all[ids_all, t, 0] < x)
            if ids.sum() > 0:
                t[ids] += 1
            else:
                break
        traj_x[:, int(x*2)] = interpolate_x_batch(traj_all, t, x)

    # resample traj given Y: 0:2:0.1
    t = torch.zeros(num_ball, dtype=torch.int64, device=device)
    for y in torch.arange(*TRAJ_Y_RANGE):
        while True:
            ids = (t < traj.shape[1] - 1) & (- traj_all[ids_all, t, 1] < y)
            if ids.sum() > 0:
                t[ids] += 1
            else:
                break
        traj_y[:, int(y*10), 0], traj_y[:, int(y*10), 1] = interpolate_y_batch(traj_all, t, -y)
    
    return traj_x.cpu().numpy(), traj_y.cpu().numpy()


class TennisBallOutEstimator():
    def __init__(self, ball_traj_out_x_file, ball_traj_out_y_file):
        self._ball_traj_out_x = np.load(ball_traj_out_x_file)
        self._ball_traj_out_y = np.load(ball_traj_out_y_file)
        self._ball_traj_out_x = torch.from_numpy(self._ball_traj_out_x)
        self._ball_traj_out_y = torch.from_numpy(self._ball_traj_out_y)
        self.params = traj_out_params
      
    def get_ball_traj_out_index(self, vel_x, vel_y, vspin):
        VEL_X_RANGE, VEL_Y_RANGE, VSPIN_RANGE = \
            self.params.VEL_X_RANGE, self.params.VEL_Y_RANGE, self.params.VSPIN_RANGE

        vel_x = torch.clamp(vel_x, VEL_X_RANGE[0], VEL_X_RANGE[1]-VEL_X_RANGE[2])
        vel_y = torch.clamp(vel_y, VEL_Y_RANGE[0], VEL_Y_RANGE[1]-VEL_Y_RANGE[2])
        vspin = torch.clamp(vspin, VSPIN_RANGE[0], VSPIN_RANGE[1]-VSPIN_RANGE[2])
        dim = (
            (VEL_X_RANGE[1] - VEL_X_RANGE[0]) / VEL_X_RANGE[2],
            (VEL_Y_RANGE[1] - VEL_Y_RANGE[0]) / VEL_Y_RANGE[2],
            (VSPIN_RANGE[1] - VSPIN_RANGE[0]) / VSPIN_RANGE[2],
        )
        index = torch.round((vel_x - VEL_X_RANGE[0]) / VEL_X_RANGE[2]) * dim[1] * dim[2] \
            + torch.round((vel_y - VEL_Y_RANGE[0]) / VEL_Y_RANGE[2]) * dim[2] \
            + torch.round((vspin - VSPIN_RANGE[0]) / VSPIN_RANGE[2])
        
        return index.cpu().long()

    def get_ball_traj_out_x_index(self, x):
        TRAJ_X_RANGE = self.params.TRAJ_X_RANGE

        x = torch.clamp(x, TRAJ_X_RANGE[0], TRAJ_X_RANGE[1]-TRAJ_X_RANGE[2])
        index = torch.round((x - TRAJ_X_RANGE[0]) / TRAJ_X_RANGE[2]) 
        return index.cpu().long()

    def get_ball_traj_out_y_index(self, y):
        TRAJ_Y_RANGE = self.params.TRAJ_Y_RANGE

        y = torch.clamp(y, TRAJ_Y_RANGE[0], TRAJ_Y_RANGE[1]-TRAJ_Y_RANGE[2])
        index = torch.round((y - TRAJ_Y_RANGE[0]) / TRAJ_Y_RANGE[2])
        return index.cpu().long()
    
    def estimate(self, ball_states_all):
        VEL_X_RANGE, VEL_Y_RANGE, TRAJ_Y_RANGE = \
            self.params.VEL_X_RANGE, self.params.VEL_Y_RANGE, self.params.TRAJ_Y_RANGE

        device = ball_states_all.get_device()
        has_valid_contact = (ball_states_all[:, 8] > VEL_X_RANGE[0]) & (ball_states_all[:, 9] > VEL_Y_RANGE[0]) \
            & (ball_states_all[:, 9] < VEL_Y_RANGE[1]) & (ball_states_all[:, 2] < TRAJ_Y_RANGE[1])
        # inside when passing the net
        x_net = ball_states_all[:, 0] + ball_states_all[:, 7] * abs(ball_states_all[:, 1] / ball_states_all[:, 8])
        has_valid_contact &= ((x_net > -4) & (x_net < 4))

        num_need_update = has_valid_contact.sum()
        if num_need_update > 0:
            ball_states = ball_states_all[has_valid_contact]
            # get index given ball state after contact
            vel_x = ball_states[:, 7:9].norm(dim=-1)
            if (vel_x >= VEL_X_RANGE[1]).sum() > 0: print('velocity X overflow')
            vel_y = ball_states[:, 9]
            vspin = ball_states[:, 10:13].norm(dim=1) / (pi * 2)
            traj_index = self.get_ball_traj_out_index(vel_x, vel_y, vspin)
            ball_traj_x = self._ball_traj_out_x[traj_index]
            ball_traj_y = self._ball_traj_out_y[traj_index]

            # update bounce pos according to the launch height
            height_index = self.get_ball_traj_out_y_index(ball_states[:, 2])
            bounce_pos = ball_states[:, :2] + \
                ball_traj_y[torch.arange(num_need_update), height_index, :1].to(device) * ball_states[:, 7:9] / vel_x.unsqueeze(-1)
            bounce_time = ball_traj_y[torch.arange(num_need_update), height_index, 1].to(device)

            # set bounce pos to 0 if ball into net
            net_dist = -ball_states[:, 1] / ball_states[:, 8] * vel_x
            net_index = self.get_ball_traj_out_x_index(net_dist)
            not_pass_net = ball_traj_x[torch.arange(num_need_update), net_index] + ball_states[:, 2].cpu() < NET_HEIGHT
            bounce_pos[not_pass_net, :] = 0
            bounce_time[not_pass_net] = 0

            # compute max ball height
            max_height = ball_states[:, 2] + ball_traj_x.max(dim=1).values.to(device)

            return has_valid_contact, bounce_pos.float(), bounce_time.float(), max_height.float()
        else:
            return has_valid_contact, None, None, None


def generate_outgoing_trajectory(traj_path, params):
    VEL_X_RANGE, VEL_Y_RANGE, VSPIN_RANGE = \
        params.VEL_X_RANGE, params.VEL_Y_RANGE, params.VSPIN_RANGE

    isaac_ball_gen = TennisBallGeneratorIsaac({}, is_train=True, need_reset=False)
    device = torch.device('cuda:0') if isaac_ball_gen.args.use_gpu_pipeline else torch.device('cpu')

    vel_y_range = np.arange(*VEL_X_RANGE)
    vel_z_range = np.arange(*VEL_Y_RANGE)
    vspin_range = np.arange(*VSPIN_RANGE)
    batch_size_tuple = (len(vel_y_range), len(vel_z_range), len(vspin_range))
    batch_size = len(vel_z_range) * len(vel_y_range) * len(vspin_range)
    batch_vel_z = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    batch_vel_y = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    batch_vspin = torch.zeros(batch_size_tuple, device=device, dtype=torch.float32)
    for i, vel_y in enumerate(vel_y_range):
        batch_vel_y[i, :, :] = vel_y
    for i, vel_z in enumerate(vel_z_range):
        batch_vel_z[:, i, :] = vel_z
    for i, vspin in enumerate(vspin_range):
        batch_vspin[:, :, i] = vspin
    batch_vel_y = batch_vel_y.view(-1)
    batch_vel_z = batch_vel_z.view(-1)
    batch_vspin = batch_vspin.view(-1)
    
    num_ball = isaac_ball_gen.num_env
    launch_pos = torch.zeros((num_ball, 3), device=device, dtype=torch.float32)
    launch_pos[:, 2] = 100
    # HACK for speed up (extremely slow if all start from the same pos)
    launch_pos[:, 0] = torch.arange(num_ball) / 1000

    traj_x_all = None
    traj_y_all = None
    for num_iter in tqdm(range(batch_size // num_ball)):
        launch_vel = torch.zeros((num_ball, 3), device=device, dtype=torch.float32)
        launch_vel[:, 1] = batch_vel_y[num_iter*num_ball: (num_iter+1)*num_ball]
        launch_vel[:, 2] = batch_vel_z[num_iter*num_ball: (num_iter+1)*num_ball]
        launch_vspin = batch_vspin[num_iter*num_ball: (num_iter+1)*num_ball]

        traj_x, traj_y = simulate_without_bounce(
            isaac_ball_gen.gym, isaac_ball_gen.sim, launch_pos.clone(), launch_vel, launch_vspin, 
            params, device=device)

        if num_iter == 0:
            traj_x_all = traj_x
            traj_y_all = traj_y
        else:
            traj_x_all = np.concatenate((traj_x_all, traj_x), axis=0)
            traj_y_all = np.concatenate((traj_y_all, traj_y), axis=0)
        np.save(traj_path, traj_x_all)
        np.save(traj_path.replace('_x', '_y'), traj_y_all)


if __name__ == '__main__':

    generate_outgoing_trajectory('data/ball_traj/ball_traj_out_x.npy', params=traj_out_params)