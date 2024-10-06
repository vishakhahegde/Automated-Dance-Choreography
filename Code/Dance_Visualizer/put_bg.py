import vedo
import torch
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import trimesh
import imageio
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest
    

def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def visualize(motion, smpl_model):
    #print("\n Inside visualise")
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    #print("\n Smpl poses shape",smpl_poses.shape)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    # print("\n Smpl poses", smpl_poses,"  shape",smpl_poses.shape)
    #print("\nSmpl poses shape",smpl_poses.shape)
    seq_len, _, _ = smpl_poses.shape
    smpl_poses = np.reshape(smpl_poses, (seq_len, 72))
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    print("Smpl trans shape",smpl_trans.shape)
    body_model_config = dict(type='smpl', model_path=smpl_model)
    visualize_smpl_pose(
        poses=smpl_poses,
        body_model_config=body_model_config,
        output_path='smpl.mp4',
        resolution=(1024, 1024))


if __name__ == "__main__":
    import glob
    import tqdm
    from smplx import SMPL

    # get cached motion features for the real data
    """real_features = {
        "kinetic": [np.load(f) for f in glob.glob("./data/aist_features/*_kinetic.npy")],
        "manual": [np.load(f) for f in glob.glob("./data/aist_features/*_manual.npy")],
    }"""

    # set smpl
    smpl = SMPL(model_path="C:/Vibha Files/PES/Capstone/Code/aistplusplus_api-main/smpl", gender='MALE', batch_size=1)

    # get motion features for the results
    result_features = {"kinetic": [], "manual": []}    
    result_motion = np.load("./outputs/esa4/gHO_sFM_cAll_d21_mHO3_ch18_mHO0_mBr1_3.npy")[None, ...] # [1, 120 + 1200, 225]
    print("motion shape", result_motion.shape)
    visualize(result_motion, smpl)
