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
    print("\n Inside visualise")
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    #print("\n Smpl poses shape",smpl_poses.shape)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    # print("\n Smpl poses", smpl_poses,"  shape",smpl_poses.shape)
    print("\nSmpl poses shape",smpl_poses.shape)
    
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    print("Smpl trans shape",smpl_trans.shape)

    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()   # (seq_len, 24, 3)

    print("Shape of final keypoints before initial",keypoints3d.shape,"\n")
    
    vertices = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
        ).vertices.detach().numpy() # first frame

    faces = smpl.faces
    print("\n Smpl trans shape",smpl_trans.shape)

    bbox_center = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        + keypoints3d.reshape(-1, 3).min(axis=0)
    ) / 2.0
    bbox_size = (
        keypoints3d.reshape(-1, 3).max(axis=0) 
        - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    vedo.show(world, axes=True, viewup="n", interactive=0)
    print("keypoints3d shape", keypoints3d.shape)
    cnt=0
 
    #output_path = 'dance_output.mp4'
    image_sequence = []
    # Define video codec and frame rate
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #fps = 60
    #out = cv2.VideoWriter(output_path, fourcc, fps, (845,960))

    for kpts in keypoints3d:
        pts = vedo.Points(kpts).c("red")
        mesh = trimesh.Trimesh(vertices[cnt], faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        #background_image = vedo.Picture("doggo.jpg")
        plotter = vedo.show(mesh, interactive=False)
        
        # Capture the current frame as an image
        img = vedo.screenshot(asarray=True)
        #print(img.shape)
        #out.write(img)
        image_sequence.append(img)
        time.sleep(0.01)
        plotter.clear()
        cnt=cnt+1

    #out.release()
    #cv2.destroyAllWindows()
    vedo.close()
    print(len(image_sequence))
    clip = ImageSequenceClip(image_sequence, fps=60)

    # Write the clip to a video filea
    output_path = 'dance_videos/esa4/esa_4_3.mp4'
    clip.write_videofile(output_path, fps=60, codec='mpeg4')


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
