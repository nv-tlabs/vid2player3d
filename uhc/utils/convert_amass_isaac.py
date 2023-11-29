import joblib
import numpy as np
import os
import sys
import argparse

sys.path.append(os.getcwd())

import torch
from scipy.spatial.transform import Rotation as sRot
import yaml
from tqdm import tqdm

from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from embodied_pose.utils.motion_lib import MotionLib
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

parser = argparse.ArgumentParser()
parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl")
parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass")
parser.add_argument('--num_seq', type=int, default=None)
parser.add_argument('--num_motion_libs', type=int, default=14)
args = parser.parse_args()

num_seq = args.num_seq
num_motion_libs = args.num_motion_libs

os.makedirs(args.out_dir, exist_ok=True)
meta_data = {
    "amass_data": args.amass_data,
    "num_seq": num_seq,
    "num_motion_libs": num_motion_libs
}
yaml.safe_dump(meta_data, open(f'{args.out_dir}/args.yml', 'w'))

amass_data = joblib.load(args.amass_data)
info = joblib.load('data/misc/smpl_body_info.pkl')

# body_shapes
all_beta = [x['beta'][:10] for x in amass_data.values()]
_, index = np.unique([",".join([f"{x:.6f}" for x in beta]) for beta in all_beta], return_index=True)
index.sort()
beta_arr = [all_beta[i] for i in index]
beta_mapping = dict()
for i, beta in enumerate(beta_arr):
    key = ",".join([f"{x:.6f}" for x in beta])
    beta_mapping[key] = i
print(f'AMASS data has {len(beta_mapping)} unique body shapes!')
joblib.dump({'beta_arr': beta_arr, 'beta_mapping': beta_mapping}, f'{args.out_dir}/shape_data.pkl')

robot_cfg = {
    "mesh": True,
    "model": "smpl",
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir= "data/smpl",
    model_xml_path=model_xml_path
)


mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_2_mujoco = [
    joint_names.index(q) for q in mujoco_joint_names
    if q in joint_names
]

amass_full_motion_dict = {}
sequences = np.array(list(amass_data.keys()))
if num_seq is not None:
    sequences = sequences[:num_seq]

seq_mapping = {seq_name.item(): seq_idx for seq_idx, seq_name in enumerate(sequences)}
motion_lib_seq_arr = np.array_split(sequences, num_motion_libs)

seq_name_splits = {}
for i, seq_arr in enumerate(motion_lib_seq_arr):
    seq_name_splits[i] = [seq_name.item()[2:] for seq_name in seq_arr]
joblib.dump(seq_name_splits, f'{args.out_dir}/seq_name_splits.pkl')


for i, motion_lib_seqs in enumerate(tqdm(motion_lib_seq_arr)):
    
    motion_lib_input_dict = dict()

    for key_name in motion_lib_seqs:
        key_name = key_name.item()
        smpl_data_entry = amass_data[key_name]
        file_name = f"data/amass/singles/{key_name}.npy"
        seq_len = smpl_data_entry['pose_aa'].shape[0]

        pose_aa = smpl_data_entry['pose_aa'].copy()
        trans = smpl_data_entry['trans'].copy()
        beta = smpl_data_entry['beta'][:10].copy()
        gender = smpl_data_entry['gender']
        fps = 30.0

        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
            smpl_parser = smpl_local_robot.smpl_parser_n
        elif gender == "male":
            gender_number = [1]
            smpl_parser = smpl_local_robot.smpl_parser_m
        elif gender == "female":
            gender_number = [2]
            smpl_parser = smpl_local_robot.smpl_parser_f
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")
        
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        root_trans = trans + skeleton_tree.local_translation[0].numpy()
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(root_trans),
            is_local=True)

        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(trans)
        )

        # min_verts_h = verts[..., 2].min().item()
        min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

        beta_key = ",".join([f"{x:.6f}" for x in beta])

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = key_name
        new_motion_out['seq_idx'] = seq_mapping[key_name]
        new_motion_out['trans'] = trans
        new_motion_out['root_trans'] = root_trans
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['beta'] = beta
        new_motion_out['beta_idx'] = beta_mapping[beta_key]
        new_motion_out['gender'] = gender
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        motion_lib_input_dict[key_name] = new_motion_out

    motion_lib = MotionLib(motion_file=motion_lib_input_dict,
        dof_body_ids=info['dof_body_ids'],
        dof_offsets=info['dof_offsets'],
        key_body_ids=info['key_body_ids'],
        device='cpu',
        clean_up=True
    )

    torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}.pth")

    del motion_lib_input_dict
    del motion_lib


smpl_local_robot.clean_up()
