from utils.pose import SMPLPose
from utils.konia_transform import angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix

import math
import torch
import numpy as np
import torch.nn.functional as F


TENNIS_RACKET_EASTERN = {
    'handle_length': 0.2, # 0.16,
    'shaft_length': 0.15, # 0.12,
    'head_radius': 0.15, # 0.26 / 2,
    'shaft_angle': np.pi / 10,

    'racket_dir':      [-1, 0, 0], # [-5, 0, 1], # 45degree [-1, 0, 1] # 90 degree [0, 0, 1]
    'racket_normal':   [0, 1, 0],
    'racket_dir_vert': [0, 0, -1],
}

TENNIS_RACKET_SEMIWESTERN = {
    'handle_length': 0.2, # 0.16,
    'shaft_length': 0.15, # 0.12,
    'head_radius': 0.15, # 0.26 / 2,
    'grip_offset': 0, # 0.04,
    'grip_handle_ratio': 0, # 0.75,
    'shaft_angle': np.pi / 10,

    'racket_dir':      [-1, 0, 0], # [-5, 0, 1], # 45degree [-1, 0, 1] # 90 degree [0, 0, 1]
    'racket_normal':   [0, 1./math.sqrt(2), 1./math.sqrt(2)],
    'racket_dir_vert': [0, 1./math.sqrt(2), -1./math.sqrt(2)],
}

TENNIS_RACKET_LEFTHAND_SEMIWESTERN = {
    'handle_length': 0.2, # 0.16,
    'shaft_length': 0.15, # 0.12,
    'head_radius': 0.15, # 0.26 / 2,
    'grip_offset': 0, # 0.04,
    'grip_handle_ratio': 0, # 0.75,
    'shaft_angle': np.pi / 10,

    'racket_dir':      [1, 0, 0], # [-5, 0, 1], # 45degree [-1, 0, 1] # 90 degree [0, 0, 1]
    'racket_normal':   [0, 1./math.sqrt(2), 1./math.sqrt(2)],
    'racket_dir_vert': [0, 1./math.sqrt(2), -1./math.sqrt(2)],
}

BADMINTON_RACKET = {
    'handle_length': 0.15,
    'shaft_length': 0.25,
    'head_radius': 0.25 / 2,
    'grip_offset': 0.02,
    'grip_handle_ratio': 0.5,

    'hand_normal':     [0, -1, 0],
    'racket_dir':      [-5, 0, 1],
    'racket_normal':   [0, 1, 0],
}

def normalize(x):
    return x / np.linalg.norm(x)

def infer_racket_from_smpl(joint_pos, joint_rot, sport='badminton', grip='eastern', righthand=True):
    # joint_pos should be relative to root
    joint_pos = joint_pos.reshape(24, 3)
    joint_rot = joint_rot.reshape(24, 3)
    if righthand:
       Hand, Wrist, Elbow, Shoulder, Collar = \
            SMPLPose.RHand, SMPLPose.RWrist, SMPLPose.RElbow, SMPLPose.RShoulder, SMPLPose.RCollar 
    else:
       Hand, Wrist, Elbow, Shoulder, Collar = \
            SMPLPose.LHand, SMPLPose.LWrist, SMPLPose.LElbow, SMPLPose.LShoulder, SMPLPose.LCollar
    
    if sport == 'tennis':
        if not righthand:
            racket_params = TENNIS_RACKET_LEFTHAND_SEMIWESTERN
        elif grip is None or grip == 'eastern':
            racket_params = TENNIS_RACKET_EASTERN
        elif grip == 'semi_western':
            racket_params = TENNIS_RACKET_SEMIWESTERN
    elif sport == 'badminton':
        racket_params = BADMINTON_RACKET

    racket_dir_canonical = normalize(np.array(racket_params['racket_dir']))
    racket_normal_canonical = normalize(np.array(racket_params['racket_normal']))
    handle_length = racket_params['handle_length']
    shaft_length = racket_params['shaft_length']
    head_radius = racket_params['head_radius']

    if sport == 'tennis':
        racket_dir_vert = normalize(np.array(racket_params['racket_dir_vert']))
        shaft_angle = racket_params['shaft_angle']
        shaft_left_dir_canonical = normalize(racket_dir_canonical / np.tan(shaft_angle) + racket_dir_vert)
        shaft_right_dir_canonical = normalize(racket_dir_canonical / np.tan(shaft_angle) - racket_dir_vert)
    hand_to_wrist = normalize(joint_pos[Wrist] - joint_pos[Hand])

    rotmat_hand = None
    for j in [SMPLPose.Pelvis, SMPLPose.Torso, SMPLPose.Spine, SMPLPose.Chest, 
        Collar, Shoulder, Elbow, Wrist]:
        rotmat = angle_axis_to_rotation_matrix(torch.from_numpy(joint_rot[j]).unsqueeze(0))[0].numpy()
        if rotmat_hand is None:
            rotmat_hand = rotmat
        else:
            rotmat_hand = rotmat_hand.dot(rotmat)
    racket_dir = rotmat_hand.dot(racket_dir_canonical)
    racket_normal = rotmat_hand.dot(racket_normal_canonical)

    if sport == 'badminton':
        grip_offset = racket_params['grip_offset']
        grip_handle_ratio = racket_params['grip_handle_ratio']
        hand_normal_canonical = normalize(np.array(racket_params['hand_normal']))
        hand_normal = rotmat_hand.dot(hand_normal_canonical)
        grip_pos = joint_pos[Hand] + hand_normal * grip_offset + hand_to_wrist * grip_offset

        handle_head = grip_pos + racket_dir * handle_length * grip_handle_ratio
        handle_end = handle_head - racket_dir * handle_length
        shaft_head = handle_head + racket_dir * shaft_length
        head_center = shaft_head + racket_dir * head_radius

        return {
            'racket_dir': racket_dir,
            'head_center': head_center,
            'racket_normal': racket_normal,
            'shaft_center': (shaft_head + handle_head) / 2,
            'handle_center': (handle_head + handle_end) / 2,
        }
    elif sport == 'tennis':
        handle_end = joint_pos[Wrist]
        handle_head = handle_end + racket_dir * handle_length
        shaft_head = handle_head + racket_dir * shaft_length
        head_center = shaft_head + racket_dir * (head_radius)
        shaft_left_dir = rotmat_hand.dot(shaft_left_dir_canonical)
        shaft_right_dir = rotmat_hand.dot(shaft_right_dir_canonical)
        shaft_left_center = handle_head + shaft_left_dir * shaft_length / np.cos(shaft_angle) / 2 
        shaft_right_center = handle_head + shaft_right_dir * shaft_length / np.cos(shaft_angle) / 2 

        return {
            'racket_dir': racket_dir,
            'head_center': head_center,
            'racket_normal': racket_normal,
            'shaft_left_center': shaft_left_center,
            'shaft_right_center': shaft_right_center,
            'shaft_left_dir': shaft_left_dir,
            'shaft_right_dir': shaft_right_dir,
            'handle_center': (handle_head + handle_end) / 2,
        }


class Racket():

    def __init__(self, sport, device, grip='eastern', righthand=True):

        if righthand:
            Hand, Wrist, Elbow, Shoulder, Collar = \
                    SMPLPose.RHand, SMPLPose.RWrist, SMPLPose.RElbow, SMPLPose.RShoulder, SMPLPose.RCollar 
        else:
            Hand, Wrist, Elbow, Shoulder, Collar = \
                    SMPLPose.LHand, SMPLPose.LWrist, SMPLPose.LElbow, SMPLPose.LShoulder, SMPLPose.LCollar
        
        # joint chain for computing racket position
        self.joints = [SMPLPose.Pelvis, SMPLPose.Torso, SMPLPose.Spine, SMPLPose.Chest, 
            Collar, Shoulder, Elbow, Wrist, Hand]

        if sport == 'tennis':
            if grip is None or grip == 'eastern':
                racket_params = TENNIS_RACKET_EASTERN
            elif grip == 'semi_western':
                racket_params = TENNIS_RACKET_SEMIWESTERN
        elif sport == 'badminton':
            racket_params = BADMINTON_RACKET        
        handle_length = racket_params['handle_length']
        shaft_length = racket_params['shaft_length']
        head_radius = racket_params['head_radius']
        grip_offset = racket_params.get('grip_offset', 0)
        grip_handle_ratio = racket_params.get('grip_handle_ratio', 0)

        racket_dir_canonical = F.normalize(torch.FloatTensor(racket_params['racket_dir']), dim=0).to(device)
        racket_normal_canonical = F.normalize(torch.FloatTensor(racket_params['racket_normal']), dim=0).to(device)
        hand_normal_canonical = F.normalize(torch.FloatTensor(racket_params.get('hand_normal', [0, -1, 0])), dim=0).to(device)

        self.params = handle_length, shaft_length, head_radius, grip_offset, grip_handle_ratio
        self.vectors = hand_normal_canonical, racket_dir_canonical, racket_normal_canonical 

    def infer_with_fk(self, joint_rotmat, joint_pos_bind_rel, root_pos):
        joints = self.joints
        hand_normal_canonical, racket_dir_canonical, racket_normal_canonical = self.vectors
        handle_length, shaft_length, head_radius, grip_offset, grip_handle_ratio = self.params

        batch_size = joint_rotmat.shape[0]
        joint_rotmat = joint_rotmat.view(batch_size, -1, 3, 3)[:, joints]
        joint_pos_bind_rel = joint_pos_bind_rel[:batch_size, joints].unsqueeze(-1)

        joint_pos_hand, joint_pos_wrist, hand_to_wrist, racket_dir, racket_normal, hand_normal = \
            infer_racket_with_fk(joint_rotmat, joint_pos_bind_rel, root_pos,
                hand_normal_canonical, racket_dir_canonical, racket_normal_canonical)

        handle_head = joint_pos_wrist + racket_dir * handle_length
        shaft_head = handle_head + racket_dir * shaft_length
        head_center = shaft_head + racket_dir * head_radius

        return {
            'pos': head_center,
            'normal': racket_normal,
        }

    def infer_without_fk(self, rb_pos, rb_rot):
        joints = self.joints
        hand_normal_canonical, racket_dir_canonical, racket_normal_canonical = self.vectors
        handle_length, shaft_length, head_radius, grip_offset, grip_handle_ratio = self.params

        rb_pos = rb_pos[:, joints]
        rb_rot = rb_rot[:, joints]
        rb_rotmat = quaternion_to_rotation_matrix(rb_rot[..., [3, 0, 1, 2]])

        joint_pos_wrist = rb_pos[:, -2]
        rotmat_wrist = rb_rotmat[:, -2]

        racket_dir = torch.matmul(rotmat_wrist, racket_dir_canonical) # N x 3
        racket_normal = torch.matmul(rotmat_wrist, racket_normal_canonical) # N x 3

        handle_head = joint_pos_wrist + racket_dir * handle_length
        shaft_head = handle_head + racket_dir * shaft_length
        head_center = shaft_head + racket_dir * head_radius

        return {
            'pos': head_center,
            'normal': racket_normal,
        }


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def infer_racket_with_fk(joint_rotmat, joint_pos_bind_rel, root_pos,
    hand_normal_canonical, racket_dir_canonical, racket_normal_canonical,
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    num_joints = joint_rotmat.shape[1]

    # creates a batch of transformation matrices
    transforms_mat = torch.cat([
            F.pad(joint_rotmat.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(joint_pos_bind_rel.reshape(-1, 3, 1), [0, 0, 0, 1], value=1.)
        ], dim=2).reshape(-1, num_joints, 4, 4)

    transform_chain = [transforms_mat[:, 0]] # Store Pelvis
    for i in range(1, num_joints):
        # subtract the joint location at the rest pose
        # no need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[i-1],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    joint_pos = transforms[:, :, :3, 3]
    joint_pos = joint_pos + root_pos[:, None, :]

    joint_pos_wrist = joint_pos[:, -2]
    joint_pos_hand = joint_pos[:, -1]
    rotmat_wrist = transforms[:, -2, :3, :3]
    hand_to_wrist = F.normalize(joint_pos_wrist - joint_pos_hand, dim=1)

    racket_dir = torch.matmul(rotmat_wrist, racket_dir_canonical) # N x 3
    racket_normal = torch.matmul(rotmat_wrist, racket_normal_canonical) # N x 3
    hand_normal = torch.matmul(rotmat_wrist, hand_normal_canonical)

    return joint_pos_hand, joint_pos_wrist, hand_to_wrist, racket_dir, racket_normal, hand_normal