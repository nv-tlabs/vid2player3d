from utils.io import *
from utils.pose import SMPLPose
from utils.common import concat
from utils.torch_transform import rotmat_to_rot6d

from torch.utils.data import Dataset
import os
import random
import copy
import torch


class Video3DPoseDataset(Dataset):

    def __init__(
        self,
        opt,
    ):
        self.opt = copy.deepcopy(opt)
        self.manifest = load_json(
            os.path.join(opt.dataset_dir, 'manifest.json'))
        self.joint_pos_arr = np.load(
            os.path.join(opt.dataset_dir, 'joint_pos.npy'), mmap_mode='r')
        self.joint_rot_arr = np.load(
            os.path.join(opt.dataset_dir, 'joint_rot.npy'), mmap_mode='r')
        if 'joint_velo' in opt.pose_feature:
            self.joint_rot_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_rot.npy'), mmap_mode='r')
        if 'joint_rotmat' in opt.pose_feature:
            self.joint_rotmat_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_rotmat.npy'), mmap_mode='r')
        if 'joint_quat' in opt.pose_feature:
            self.joint_quat_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_quat.npy'), mmap_mode='r')
        self.valid_arr = np.load(
            os.path.join(opt.dataset_dir, 'valid.npy'))

        self.sequences = []
        self.selected_arr = np.zeros_like(self.valid_arr)
        if opt.predict_phase:
            self.phase_arr = np.zeros((self.valid_arr.shape[0], 2), dtype=float)
            self.phase_rad_arr = np.zeros(self.valid_arr.shape[0], dtype=float)
        
        def find_neighboring_hits(point, fid):
            assert fid <= point[-1]['fid']
            for hid, hit in enumerate(point):
                if fid == point[hid+1]['fid'] and hid == 0:
                    return point[hid+1], point[hid+2]
                if fid >= hit['fid'] and fid <= point[hid+1]['fid']: 
                    return hit, point[hid+1]
        
        num_videos = 0
        betas = []
        # Fliter sequences
        for video in self.manifest:
            if video['background'] not in opt.background: continue
            if video['gender'] not in opt.gender: continue
            if opt.sport == 'tennis':
                if video.get('is_orig') == True and 'orig' not in opt.split_annotation: continue
                if video.get('is_orig') == False and 'weak' not in opt.split_annotation: continue
            print(video['name'])

            if opt.side == 'both':
                seqs_candidate = video['sequences']['fg'] + video['sequences']['bg']
            elif opt.side == 'fg':
                seqs_candidate = video['sequences']['fg']
            elif opt.side == 'bg':
                seqs_candidate = video['sequences']['bg']

            num_videos += 1
            for seq in seqs_candidate:
                if opt.player_handness is not None:
                    if seq['handness'] not in opt.player_handness: continue
                else:
                    if video.get('is_orig') or seq['player'] is not None:
                        # These videos have player identity annotated
                        if opt.player_name is not None and seq['player'] not in opt.player_name: continue
                seq = seq.copy()
                
                if opt.predict_phase:
                    if not video.get('is_orig'): continue
                    point = video['points_annotation'][seq['point_idx']]['keyframes']
                    for idx in range(seq['length']):
                        fid = idx + seq['start']
                        arr_idx = idx + seq['base']
                        prev_hit, next_hit = find_neighboring_hits(point, fid)
                        phase = (fid - prev_hit['fid']) / (next_hit['fid'] - prev_hit['fid'])
                        assert opt.side == 'fg'
                        phase += 1 if prev_hit['fg'] else 0 # add 1 if in recovery
                        self.phase_arr[arr_idx] = np.array([np.sin(phase * np.pi), np.cos(phase * np.pi)])
                        self.phase_rad_arr[arr_idx] = phase * np.pi
                    seq['has_phase'] = True 
                
                self.sequences += [seq]
                self.selected_arr[seq['base'] : seq['base']+seq['length']] = 1
                betas += [seq['beta']]
              
        num_valid_frames = np.logical_and(self.valid_arr, self.selected_arr).sum()
        print(f"Loaded {len(self.sequences)} motion sequences from {num_videos} videos containing {num_valid_frames} frames")

        self.std, self.avg = None, None

        self.seq_weights = np.array(
            [seq['length'] for seq in self.sequences], dtype=float)
        self.seq_weights /= np.sum(self.seq_weights)

        self.init_rollouts(opt.nframes_seq)


    def init_rollouts(self, nframes_seq):
        self.nframes_seq = nframes_seq # nframes_seq might be different from opt.nframes_seq
        self.rollouts = []
        if self.opt.database_ratio != 1.0:
            total_seqs = int(len(self.sequences) * self.opt.database_ratio)
            self.sequences = self.sequences[:total_seqs]
        for seq in self.sequences:
            for x in range(seq['base'], seq['base'] + seq['length'] - nframes_seq - 1):
                if self.valid_arr[x : x + nframes_seq + 1].sum() == nframes_seq + 1:
                    self.rollouts += [x]
        
        random.shuffle(self.rollouts)
        print("Init {} rollouts".format(len(self.rollouts)))
    

    def get_normalization_stats(self):
        opt = self.opt
        feature_all = None
        if 'root_pos' in opt.pose_feature:
            root_pos = self.joint_pos_arr[:, :3]
            if opt.condition_root_x_only:
                root_pos = root_pos[:, 0:1]
            elif opt.no_condition_root_y:
                root_pos = root_pos[:, [0, 2]]
            feature_all = concat(feature_all, root_pos, axis=1)
        if 'root_velo' in opt.pose_feature:
            root_velo = self.joint_pos_arr[:, :3] - np.roll(self.joint_pos_arr[:, :3], 1, axis=0)
            feature_all = concat(feature_all, root_velo, axis=1)
        if 'joint_pos' in opt.pose_feature:
            joint_pos = self.joint_pos_arr[:, 3:]
            feature_all = concat(feature_all, joint_pos, axis=1)
        if 'joint_velo' in opt.pose_feature:
            joint_velo = self.joint_pos_arr[:, 3:] - np.roll(self.joint_pos_arr[:, 3:], 1, axis=0)
            feature_all = concat(feature_all, joint_velo, axis=1)
        if 'joint_quat' in opt.pose_feature:
            joint_quat = self.joint_quat_arr.reshape(-1, 24*4)
            feature_all = concat(feature_all, joint_quat, axis=1)
        if 'joint_rotmat' in opt.pose_feature:
            joint_rotmat = rotmat_to_rot6d(torch.from_numpy(
                self.joint_rotmat_arr.reshape(-1, 24, 3, 3).copy())).numpy().reshape(-1, 24*6)
            feature_all = concat(feature_all, joint_rotmat, axis=1)

        feature_all = feature_all[np.logical_and(self.valid_arr, self.selected_arr)]

        std = np.std(feature_all, axis=0)
        std[std == 0] = 1.0
        avg = np.average(feature_all, axis=0)
        self.std = std
        self.avg = avg
    
    
    def set_normalization_stats(self, avg, std):
        self.avg = avg
        self.std = std


    def __len__(self):
        return self.opt.nseqs


    def __getitem__(self, idx):
        # Required
        # root velocity 3: do we need to share the same coordinate of the last frame
        # joint position 23 x 3
        # joint velocity 23 x 3: do we need this if we are considering more than one condition frames
        # root and joint orientation 24 x 6 first two columns of the rotation matrix 

        # Optional
        # joint angular velocity 24 x 3
        # global root position 3
        # contact  
        opt = self.opt
        if opt.test_only and opt.batch_size <= len(self.rollouts):
            start = self.rollouts[idx]
        else:
            start = random.choices(self.rollouts, k=1)[0]
        end = start + self.nframes_seq + 1
        L = self.nframes_seq

        feature = None
        # the actual frame starts at start + 1
        if 'root_pos' in opt.pose_feature:
            root_pos = self.joint_pos_arr[start+1:end, :3]
            if opt.condition_root_x_only:
                root_pos = root_pos[:, 0:1]
            elif opt.no_condition_root_y:
                root_pos = root_pos[:, [0, 2]]
            feature = concat(feature, root_pos, axis=1)
        if 'root_velo' in opt.pose_feature:
            root_velo = self.joint_pos_arr[start+1:end, :3] - self.joint_pos_arr[start:end-1, :3]
            feature = concat(feature, root_velo, axis=1)
        if 'joint_pos' in opt.pose_feature:
            joint_pos = self.joint_pos_arr[start+1:end, 3:]
            feature = concat(feature, joint_pos, axis=1)
        if 'joint_velo' in opt.pose_feature:
            joint_velo = self.joint_pos_arr[start+1:end, 3:] - self.joint_pos_arr[start:end-1, 3:]
            feature = concat(feature, joint_velo, axis=1)
        if 'joint_rotmat' in opt.pose_feature:
            joint_rotmat = rotmat_to_rot6d(torch.from_numpy(
                self.joint_rotmat_arr[start+1:end].reshape(L, 24, 3, 3).copy())).numpy().reshape(L, 24*6)
            feature = concat(feature, joint_rotmat, axis=1)
        if 'joint_quat' in opt.pose_feature:
            joint_quat = self.joint_quat_arr[start+1:end].reshape(-1, 24*4)
            feature = concat(feature, joint_quat, axis=1)

        if self.std is not None:
            feature = (feature - self.avg) / self.std
        data_dict = {
            'feature': feature,
            'start': start,
        }
        if opt.predict_phase:
            data_dict['phase'] = self.phase_arr[start+1:end]
        return data_dict
    

    def sample_first_frame(self):
        opt = self.opt
        data = self.__getitem__(0)
        T = self.opt.num_condition_frames
        # the actual frame starts at data['start'] + 1
        frame = {
            'root_pos': self.joint_pos_arr[data['start'] + T, :3].copy(),
            'joint_pos': self.joint_pos_arr[data['start'] + T, 3:].copy(),
            'joint_rot': self.joint_rot_arr[data['start'] + T].copy(),
            'condition': data['feature'][:T]
        }
        return frame