import os
import argparse
import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R
from typing import List, Union


# dataset_rotate = {'KIT': [0, 0, 90],   # -x
#                 'Eyes_Japan_Dataset': 1, 'SSM_synced': 1, 'BMLmovi': 1,
#                 'TotalCapture': 1, 'MPI_HDM05': 1, 'MPI_mosh': 1, 'EKUT': 1, 'Transitions_mocap': 1, 'DFaust_67': 1,
#                 'ACCAD': 1, 'BioMotionLab_NTroje': 1, 'MPI_Limits': 1, 'SFU': 1, 'BMLhandball': 1, 'HumanEva': 1}


def smplx_angles_to_rotate(z_angle, x_angle, threshold=2.5):
    if 45 + threshold < z_angle <= 135 - threshold:
        return [0, 0, -90]
    if -135 + threshold < z_angle <= -45 - threshold:
        return [0, 0, 90]
    if 135 + threshold < x_angle <= 180 or -180 <= x_angle <= -135 - threshold:
        return [0, 0, 180]
    if 135 + threshold < z_angle <= 180 or -180 <= z_angle <= -135 - threshold:
        return [0, 0, 180]
    if -45 + threshold < z_angle <= 45 - threshold:
        return [0, 0, 0]
    return [-1, -1, -1]



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_HumanML3D_motion', type=str, default="/ceph/datasets/SMPLX/HumanML3D/motion_data/processed/")
    parser.add_argument('--processed_motion', type=str, default="/ceph/datasets/SMPLX/HumanML3D/processed_motion/")
    parser.add_argument('--index_path', type=str, default="/ceph/hdd/yangsc21/Python/mdm/prepare/index.csv")
    args = parser.parse_args()
    return args


def save_npz(output_file, smplx_trans, smplx_poses, gender='neutral', model_type='smplx', frame_rate=30):
    np.savez(output_file, trans=smplx_trans, poses=smplx_poses, gender=gender, surface_model_type=model_type,
             mocap_frame_rate=frame_rate, betas=np.zeros(16))


def rotate_smplx_root(smplx_poses: np.ndarray, smplx_trans: np.ndarray, rotxyz: Union[np.ndarray, List]):
    if rotxyz == [0, 0, 0]:
        return np.copy(smplx_poses), np.copy(smplx_trans)
    if rotxyz == [-1, -1, -1]:
        return None, None
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    transformation_euler = np.deg2rad(rotxyz)
    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)

    # 应用变换到根方向
    root_orient = smplx_poses[:, :3]

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()

    # 应用变换到平移向量
    rotated_trans = np.matmul(coord_change_matrot, smplx_trans.T).T

    # 组合回poses数组
    rotated_poses = np.copy(smplx_poses)
    rotated_poses[:, :3] = transformed_root_orient

    return rotated_poses, rotated_trans


def get_average_smplx_orientation(smplx_poses: np.ndarray) -> np.ndarray:
    """
    Get the average orientation of SMPLX model throughout the motion sequence
    using vector averaging to handle circular nature of angles.

    Parameters
    ----------
    smplx_poses: np.ndarray
        Array of SMPLX poses with shape (t, 165).

    Returns
    -------
    np.ndarray
        Euler angles representing the average orientation of the model.
    """
    root_orient = smplx_poses[:10, :3]
    euler_angles = np.array([R.from_rotvec(rot_vec).as_euler('XYZ', degrees=True) for rot_vec in root_orient])

    # 转换角度为二维向量
    vectors = np.array([np.cos(np.deg2rad(euler_angles[:, 2])), np.sin(np.deg2rad(euler_angles[:, 2]))]).T

    # 计算向量的平均值
    mean_vector = np.mean(vectors, axis=0)

    # 将平均向量转换回角度
    average_angle = np.arctan2(mean_vector[1], mean_vector[0])
    average_euler_angles = np.mean(euler_angles, axis=0)
    average_euler_angles[2] = np.rad2deg(average_angle)

    return average_euler_angles


def process_index(file_path):
    # ./pose_data/MPI_HDM05/bk/HDM_bk_02-03_02_120_poses.npy,2,202,005553.npy
    index_dict = {}
    dataset_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            source_path, start_frame, end_frame, new_name = line.split(',')
            name = new_name.split('.')[0]
            dataset = source_path.split('/')[2]
            index_dict[name] = [dataset, int(start_frame), int(end_frame)]
            if dataset not in dataset_dict.keys():
                dataset_dict[dataset] = 1
    return index_dict, dataset_dict


# We noticed that the orientation of the HDM05 dataset is not consistent,
# e. g. SMPLX-neutral_HDM_mm_03-10_07_120_stageii, SMPLX-neutral_HDM_bk_02-03_02_120_stageii ...
# SMPLX-neutral_93_07_stageii, SMPLX-neutral_75_11_stageii, SMPLX-neutral_60_10_stageii

def process_motion(input_path, output_path, dict_index):

    error_list = []

    for file in sorted(os.listdir(input_path)):
    # for file in ['005060.npz', '006023.npz', '012366.npz', '014030.npz']:       # debug
    # for file in ['013954.npz']:
        name, _ = os.path.splitext(file)
        # print('processing', name, 'error', error_list)
        print('processing', name)
        output_file = os.path.join(output_path, file)

        if os.path.exists(output_file):
            print('Already exists, skip')
            continue

        smplx_motion = np.load(os.path.join(input_path, file))
        # ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'markers_latent', 'latent_labels',
        # 'markers_latent_vids', 'trans', 'poses', 'betas', 'num_betas', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
        smplx_fps = smplx_motion['mocap_frame_rate']
        smplx_poses = smplx_motion['poses']
        smplx_trans = smplx_motion['trans']
        # smplx_mocap_time_length = smplx_motion['mocap_time_length']     # 5.82


        # if int(smplx_fps) % 20 != 0:
        if np.around(smplx_fps) % 20 != 0:
            # raise Exception('fps is {}'.format(smplx_fps))
            error_list.append(name)
            print('error', name, 'fps is {}'.format(smplx_fps))
        else:

            dataset, start_frame, end_frame = dict_index[name]
            if dataset == 'humanact12':
                continue

            smplx_fps = int(smplx_fps)
            divided = int(smplx_fps / 20)

            smplx_poses = smplx_poses[::divided]
            smplx_trans = smplx_trans[::divided]
            smplx_poses = smplx_poses[start_frame:end_frame]
            smplx_trans = smplx_trans[start_frame:end_frame]

            try:
            # if True:
                begin_position = smplx_trans[0]
                smplx_trans = smplx_trans - begin_position

                smplx_angles = get_average_smplx_orientation(smplx_poses)

                # print('smplx_angles', smplx_angles)

                smplx_poses, smplx_trans = rotate_smplx_root(smplx_poses, smplx_trans, smplx_angles_to_rotate(smplx_angles[1], smplx_angles[2]))
                # smplx_angles = get_average_smplx_orientation(smplx_poses)
                # print('smplx_angles', smplx_angles)
                if not isinstance(smplx_poses, np.ndarray):
                    print('threshold of smplx_angles[1] is under 2.5')
                    raise Exception
                save_npz(output_file, smplx_trans, smplx_poses, frame_rate=20)

            except:
                error_list.append(name)
                print('error', name)


    print('error_list', error_list)


if __name__ == '__main__':
    '''
python process_amass.py --source_HumanML3D_motion /ceph/datasets/SMPLX/HumanML3D/motion_data/processed --processed_motion /ceph/datasets/SMPLX/HumanML3D/processed_motion/ --index_path ../prepare/index.csv
    '''
    args = get_args()

    source_HumanML3D_motion = args.source_HumanML3D_motion

    processed_motion = args.processed_motion

    if not os.path.exists(processed_motion):
        os.makedirs(processed_motion)

    dict_index, dataset_dict = process_index(args.index_path)
    # print('dataset_dict', dataset_dict.keys())
    process_motion(source_HumanML3D_motion, processed_motion, dict_index)
