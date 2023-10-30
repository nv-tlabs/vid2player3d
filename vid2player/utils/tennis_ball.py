from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

# from .io import * 
from smpl_visualizer.vis_sport import SportVisualizer

import numpy as np
from math import sqrt, pi
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

m   = 0.057 # kg, weight of ball
R   = 0.032 # m, radius of ball
rho = 1.21  # kg/m^3, density of air
g   = 9.81  # m/s^2, gravitational field
fps = 30
NET_HEIGHT = 1.07 # m
kv   = (rho * pi * R * R) / (2 * m)
kf   = (rho * pi * R * R) / 2

DT  = 1 / 1000
BASE_CD = 0.55 # physics.usyd.edu.au/~cross/TRAJECTORIES/42.%20Ball%20Trajectories.pdf
BASE_COR = 0.75 # tennisindustrymag.com/articles/2004/04/follow_the_bouncing_ball.html
BASE_COF = 0.6


def get_cl(v, vs):
    """ Coefficient of lift, magnus effect """
    return 1 / (2 + abs(v/(vs + 1e-6)))


def get_cd(v, vs):
    """ Coefficient of drag, affected by magnus. """
    return BASE_CD


def torch_sample_range(size, min, max):
    return torch.rand(size) * (max - min) + min


def get_sim_params(substeps=2):
    # parse arguments
    args = gymutil.parse_arguments(
        description="Collision Filtering: Demonstrates filtering of collisions within and between environments", 
        custom_parameters=[
        {"name": "--enable_gym_viewer", "action": "store_true", "help": "enable gym viewer"},
        ])
    
    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1. / 60.
    sim_params.substeps = substeps
    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.shape_collision_margin = 0.25
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4 if substeps == 2 else 2
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 16

        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 10.0

        sim_params.physx.use_gpu = args.use_gpu_pipeline
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # change world coord
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity.x = 0
    sim_params.gravity.y = 0
    sim_params.gravity.z = -9.81

    return sim_params, args


def create_sim(sim_params, compute_device_id, graphics_device_id, physics_engine):
    # initialize gym
    gym = gymapi.acquire_gym()

    sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.5
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create viewer
    # viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # if viewer is None:
    #     print("*** Failed to create viewer")
    #     quit()

    # subscribe to spacebar event for reset
    # gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

    return gym, sim


def simulate(gym, sim, launch_pos, launch_vel, launch_vspin, 
    control_freq_inv=2, num_frames=100, substeps=6, spin_scale=5,
    enable_gym_viewer=False, device=torch.device('cpu')):

    if enable_gym_viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()
        # gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, -20, 3), gymapi.Vec3(0, 0, 0))
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(15, -6, 3), gymapi.Vec3(0, -6, 0))
        
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
    root_state[:num_ball, 0:3] = launch_pos
    root_state[:num_ball, 7:10] = launch_vel
    root_state[:num_ball, 10:13] = launch_ang_vel
    launch_vspin = launch_vspin.clone()

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

    traj = None
    bounce_pos = torch.zeros((num_env, 3), device=device, dtype=torch.float32)
    bounce_idx = torch.zeros(num_env, device=device, dtype=torch.int64) + num_frames - 1
    has_bounce = torch.zeros(num_env, device=device, dtype=torch.bool)
    has_pass_net = torch.zeros(num_env, device=device, dtype=torch.bool)
    pass_net_success = torch.zeros(num_env, device=device, dtype=torch.bool)
    for t in range(num_frames):
        pos = root_state[:, :3].view(-1, 1, 3)
        if traj is None:
            traj = pos[:num_ball].clone()
        else:
            traj = torch.cat((traj, pos[:num_ball]), dim=1)

        for i in range(control_freq_inv):
            pos = root_state[:, 0:3]
            vel = root_state[:, 7:10]
            vel_scalar = vel.norm(dim=1).view(-1, 1)
            vel_norm = vel / vel_scalar
            vel_tan = torch.cross(vel_norm, g_tensor)
            vspin = root_state[:, 10:13].norm(dim=1) / (pi * 2)
            vspin[:num_ball] = torch.where(launch_vspin > 0, vspin[:num_ball], vspin[:num_ball] * -1)
            vspin = vspin.view(-1, 1)

            pass_net_now = (~has_pass_net) & (pos[:, 1] < 0)
            pass_net_success[pass_net_now] = (~has_bounce[pass_net_now]) & (pos[pass_net_now, 2] > NET_HEIGHT)
            has_pass_net = has_pass_net | pass_net_now

            cd = get_cd(vel_scalar, vspin * spin_scale)
            cl = get_cl(vel_scalar, vspin * spin_scale)
            cl = cl * torch.where(vspin > 0, 
                -1 * torch.ones_like(cl),
                1 * torch.ones_like(cl)
            )

            force_drag = - kf * cd * vel_scalar * vel
            force_lift = - kf * cl * vel_scalar ** 2 * torch.cross(vel_tan, vel_norm) 
            
            if substeps > 2:
                has_bounce_now = ~has_bounce & (pos[:, 2] <= R * 6)
            else:
                has_bounce_now = ~has_bounce & (pos[:, 2] <= R * 4)
            bounce_pos[has_bounce_now] = pos[has_bounce_now].clone()
            bounce_idx[has_bounce_now] = t
            has_bounce = has_bounce | has_bounce_now
            forces = force_drag + force_lift
           
            # Hack: backspin ball changes to topspin after bounce
            launch_vspin[has_bounce_now[:num_ball]] = torch.where(
                launch_vspin[has_bounce_now[:num_ball]] > 0,
                launch_vspin[has_bounce_now[:num_ball]],
                launch_vspin[has_bounce_now[:num_ball]] * -1,
            )

            forces[num_ball:, :] = 0
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), 
                None, gymapi.ENV_SPACE)

            if enable_gym_viewer:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
            
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            if enable_gym_viewer:
                gym.sync_frame_time(sim)
    
    if enable_gym_viewer:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

    return traj, bounce_pos[:num_ball], bounce_idx[:num_ball], pass_net_success[:num_ball]


class TennisBallGeneratorIsaac():
    def __init__(self, cfg, is_train=True, need_traj=True, need_reset=True, substeps=6, spin_scale=5):
        
        self.is_train = is_train
        self.need_traj = need_traj
        self.substeps = substeps
        self.spin_scale = spin_scale
        self.traj_pool = None

        sim_params, args = get_sim_params(substeps)
        gym, sim = create_sim(sim_params, args.compute_device_id, args.graphics_device_id, args.physics_engine)

        self.sim_params = sim_params
        self.args = args
        self.gym = gym
        self.sim = sim
        self.device = torch.device('cuda:0') if args.use_gpu_pipeline else torch.device('cpu')

        # load ball asset
        asset_root = "vid2player/data/assets"
        asset_file = "tennis_ball.urdf"
        asset_options = gymapi.AssetOptions()
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        self.num_env = 10000 if is_train else 1000
        num_per_row = int(np.sqrt(self.num_env))
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs = []
        for i in range(self.num_env):
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            self.envs += [env]

            # create ball
            pose = gymapi.Transform()
            pose.r = gymapi.Quat(0, 0, 0, 1)
            pose.p = gymapi.Vec3(0, 0, 1)

            # Nothing should collide.
            collision_group = i
            collision_filter = 1
            
            ahandle = gym.create_actor(env, asset, pose, None, collision_group, collision_filter)

            props = gym.get_actor_rigid_shape_properties(env, ahandle)
            props[0].restitution = 0.9
            props[0].friction = 0.2
            props[0].compliance = 0.5
            gym.set_actor_rigid_shape_properties(env, ahandle, props)
            
            gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        
        gym.prepare_sim(sim)

        self.traj_length = cfg.get('ball_traj_length', 100)
        self.origin_min = torch.FloatTensor(cfg.get('origin_min', [-4, 12, 1]))
        self.origin_max = torch.FloatTensor(cfg.get('origin_max', [4, 13, 1.5]))
        self.bounce_min = torch.FloatTensor(cfg.get('bounce_min', [-3, -10, 0]))
        self.bounce_max = torch.FloatTensor(cfg.get('bounce_max', [3, -7, 0]))
        self.vel_range = torch.FloatTensor(cfg.get('vel_range', [28, 30]))
        self.vspin_range = torch.FloatTensor(cfg.get('vspin_range', [5, 10]))
        self.theta_range = torch.FloatTensor(cfg.get('theta_range', [5, 15]))

        if need_reset:
            self.reset()
    
    def reset(self):
        num_samples = self.num_env

        origin = torch_sample_range((num_samples, 3), self.origin_min, self.origin_max)
        bounce = torch_sample_range((num_samples, 3), self.bounce_min, self.bounce_max)
        dir = F.normalize(bounce[:, :2] - origin[:, :2], dim=1)

        launch_vel_scalar = torch_sample_range((num_samples), self.vel_range[0], self.vel_range[1])
        launch_theta = torch_sample_range((num_samples), self.theta_range[0], self.theta_range[1])
        launch_vspin = torch_sample_range((num_samples), self.vspin_range[0], self.vspin_range[1])

        launch_pos = origin.to(self.device)
        launch_vel = torch.stack([
            launch_vel_scalar * torch.cos(launch_theta / 180 * np.pi) * dir[:, 0],
            launch_vel_scalar * torch.cos(launch_theta / 180 * np.pi) * dir[:, 1],
            launch_vel_scalar * torch.sin(launch_theta / 180 * np.pi),
        ]).T.to(self.device)
        launch_vspin = launch_vspin.to(self.device)

        traj, bounce_pos, bounce_idx, pass_net = simulate(
            self.gym, self.sim, launch_pos, launch_vel, launch_vspin, 
            substeps=self.substeps, spin_scale=self.spin_scale,
            device=self.device)
        
        # good trajectory has to
        # 1. pass net
        # 2. bounce inside otherside of court
        # 3. first bounce after 45 frames 
        # 4. bounce higher than 1m

        valid = pass_net.bool() & \
            (bounce_pos.sum() != 0) & \
            (bounce_pos[:, 0] > self.bounce_min[0]) & \
            (bounce_pos[:, 0] < self.bounce_max[0]) & \
            (bounce_pos[:, 1] > self.bounce_min[1]) & \
            (bounce_pos[:, 1] < self.bounce_max[1])
        
        for i in range(self.num_env):
            valid[i] &= traj[i, bounce_idx[i]:, 2].max() > 1.0

        print("Generated {}/{} valid ball trajectories".format(valid.sum(), num_samples))
        assert valid.sum() > 0

        self.traj_pool = traj[valid].cpu()
        self.launch_pos = launch_pos[valid].cpu()
        self.launch_vel = launch_vel[valid].cpu()
        self.launch_vspin = launch_vspin[valid].cpu()

    def generate(self, n_traj, need_init_state=False):
        indices = torch.randint(0, len(self.traj_pool), (n_traj,))
        selected_traj = self.traj_pool[indices].clone()
        if need_init_state:
            launch_pos = self.launch_pos[indices]
            launch_vel = self.launch_vel[indices]
            launch_vspin = self.launch_vspin[indices]
            return selected_traj, launch_pos, launch_vel, launch_vspin
        else:
            return selected_traj

    def generate_init_state(self, n_ball):
        indices = torch.randint(0, len(self.launch_pos), (n_ball,))
        launch_pos = self.launch_pos[indices]
        launch_vel = self.launch_vel[indices]
        launch_vspin = self.launch_vspin[indices]
        return launch_pos, launch_vel, launch_vspin
    
    def generate_all(self):
        return self.traj_pool, self.launch_pos, self.launch_vel, self.launch_vspin
    

def generate_incoming_trajectory(traj_path, substeps, vis=False, append=True):
    if vis:
        visualizer = SportVisualizer(verbose=False, 
            show_smpl=False, show_skeleton=False, show_racket=False,
            show_target=False, show_ball=True, track_first_actor=False)

    if os.path.exists(traj_path) and append:
        traj_data_all = np.load(traj_path)
        traj_data_all = torch.from_numpy(traj_data_all)
        print(traj_data_all.shape)
    else:
        traj_data_all = None
    isaac_ball_gen = TennisBallGeneratorIsaac({}, is_train=True, need_reset=False, substeps=substeps)
    for i in tqdm(range(100)):
        isaac_ball_gen.reset()
        try:
            traj, launch_pos, launch_vel, launch_vspin = isaac_ball_gen.generate_all()
        except Exception:
            continue
        traj_data = torch.cat([launch_pos, launch_vel, launch_vspin.view(-1, 1), traj.view(-1, 300)], dim=1)
        if traj_data_all is None:
            traj_data_all = traj_data
        else:
            traj_data_all = torch.cat([traj_data_all, traj_data], dim=0)

        np.save(traj_path, traj_data_all.numpy())
        print(traj_data_all.shape)
        if traj_data_all.shape[0] > 1000000: break
        if vis: break
    
    # Sort the traj by launch x
    traj_data_all = traj_data_all.numpy()
    traj_x = traj_data_all[:, 0]
    reorder = np.argsort(traj_x)
    traj_data_sort = traj_data_all[reorder]
    np.save(traj_path, traj_data_sort)
    
    if vis:
        traj = traj[:5]
        ball_seq = []
        for j in range(traj.shape[0]):
            ball_seq_one = []
            for i in range(traj.shape[1]):
                ball_seq_one += [{
                    'pos': traj[j][i],
                }]
            ball_seq += [ball_seq_one]

        init_args = {
            'num_actors': traj.shape[0], 
            'sport': 'tennis',
            'camera': 'front',
            'ball_seq': ball_seq,
        }

        visualizer.show_animation(
                    init_args=init_args, 
                    fps=30, 
                    window_size=(1920, 1080),
                    repeat=True,
                )


class TennisBallGeneratorOffline():
    def __init__(self, traj_file, sample_random=False, num_envs=None):
        traj_data_all = np.load(traj_file)
        traj_data_all = torch.from_numpy(traj_data_all)

        self.launch_pos = traj_data_all[:, 0:3]
        self.launch_vel = traj_data_all[:, 3:6]
        self.launch_vspin = traj_data_all[:, 6]
        self.traj_pool = traj_data_all[:, 7:].view(-1, 100, 3)
        self.sample_random = sample_random
        if not self.sample_random:
            self.sample_idx = torch.zeros((num_envs), dtype=torch.int64)
    
    def generate(self, n_traj, need_init_state=False, start_pos=None, env_ids=None):
        if self.sample_random:
            indices = torch.randint(0, len(self.traj_pool), (n_traj,))
            if start_pos is not None:
                other_side = start_pos[:, 1] > 0
                if other_side.sum() > 0:
                    indices[other_side] = ((start_pos[other_side, 0] + 4) / 8 * len(self.traj_pool)).long()
                    indices[other_side] += torch.randint(-1000, 1000, (other_side.sum(),))
                    indices = torch.clamp(indices, 0, len(self.traj_pool)-1)
        else:
            indices = self.sample_idx[env_ids]
            self.sample_idx[env_ids] += 1
            self.sample_idx[env_ids] %= len(self.traj_pool)

        selected_traj = self.traj_pool[indices].clone()
        if need_init_state:
            launch_pos = self.launch_pos[indices]
            launch_vel = self.launch_vel[indices]
            launch_vspin = self.launch_vspin[indices]
            return selected_traj, launch_pos, launch_vel, launch_vspin
        else:
            return selected_traj
        

if __name__ == '__main__':

    generate_incoming_trajectory(
        'data/ball_traj/ball_traj_in.npy', substeps=6, vis=False, append=False)