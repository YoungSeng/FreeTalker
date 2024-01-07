import numpy as np
import pdb
from animation import plot_3d_motion
import os
from skeleton import *
import argparse


kinematic_chain = [[0, 2, 5, 8, 11],  # 右侧下肢
                   [0, 1, 4, 7, 10],  # 左侧下肢
                   [0, 3, 6, 9, 12, 15],  # 脊柱和头部
                   [9, 14, 17, 19, 21],  # 右侧上肢
                   [9, 13, 16, 18, 20],  # 左侧上肢
                   # [15, 22],  # 颌部
                   # [15, 23],  # 左眼
                   # [15, 24],  # 右眼
                   [21, 40, 41, 42],  # 右手食指
                   [21, 43, 44, 45],  # 右手中指
                   [21, 46, 47, 48],  # 右手小指
                   [21, 49, 50, 51],  # 右手无名指
                   [21, 52, 53, 54],  # 右手拇指
                   [20, 25, 26, 27],  # 左手食指
                   [20, 28, 29, 30],  # 左手中指
                   [20, 31, 32, 33],  # 左手小指
                   [20, 34, 35, 36],  # 左手无名指
                   [20, 37, 38, 39],  # 左手拇指
                   ]

# pelvis关节是根关节，它的位置是[0, 0, 0]。left_hip关节相对于pelvis关节的位置是[1, 0, 0]，表示left_hip关节在pelvis关节的x
# 轴正方向上有一个单位的距离。类似地，right_hip关节相对于pelvis关节的位置是[-1, 0, 0]，表示right_hip关节在pelvis关节的
# x轴负方向上有一个单位的距离
# SMPL-X 模型的确有 127 个关节，但并非所有关节都是独立可控制的。实际上，SMPL-X 模型有 55 个独立可控制的关节。这些关节主要用于描述人体的基本姿势和形状，如躯干、四肢、手指和脚趾等。
# 另外，剩余的关节可能是用于辅助建模的，例如用于实现更精细的动画效果、碰撞检测或者其他特定功能。这些关节可能不会直接用于控制人体姿势，但在模型的构建和渲染过程中起到了重要作用。

n_raw_offsets = torch.tensor([[0, 0, 0],  # 0: 'pelvis',
                              [1, 0, 0],  # 1: 'left_hip',
                              [-1, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0],
                              [0, -1, 0],
                              [0, 1, 0],
                              [0, -1, 0],
                              [0, -1, 0],
                              [0, 1, 0],
                              [0, 0, 1],  # 10: 'left_foot',
                              [0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0],
                              [-1, 0, 0],
                              [0, 0, 1],  # 15: 'head',
                              [0, -1, 0],
                              [0, -1, 0],
                              [0, -1, 0],  # 18: 'left_elbow',
                              [0, -1, 0],  # 19: 'right_elbow',
                              [0, -1, 0],  # 20: 'left_wrist',
                              [0, -1, 0],  # 21: 'right_wrist',
                              [0, -1, 0],  # 22: 'jaw',
                              [1, 0, 0],  # 23: 'left_eye',
                              [-1, 0, 0],  # 24: 'right_eye',
                              [0, -1, 0],  # 25: 'left_index1',
                              [0, -1, 0],  # 26: 'left_index2',
                              [0, -1, 0],  # 27: 'left_index3',
                              [0, -1, 0],  # 28: 'left_middle1',
                              [0, -1, 0],  # 29: 'left_middle2',
                              [0, -1, 0],  # 30: 'left_middle3',
                              [0, -1, 0],  # 31: 'left_pinky1',
                              [0, -1, 0],  # 32: 'left_pinky2',
                              [0, -1, 0],  # 33: 'left_pinky3',
                              [0, -1, 0],  # 34: 'left_ring1',
                              [0, -1, 0],  # 35: 'left_ring2',
                              [0, -1, 0],  # 36: 'left_ring3',
                              [0, -1, 0],  # 37: 'left_thumb1',
                              [0, -1, 0],  # 38: 'left_thumb2',
                              [0, -1, 0],  # 39: 'left_thumb3',
                              [0, -1, 0],  # 40: 'right_index1',
                              [0, -1, 0],  # 41: 'right_index2',
                              [0, -1, 0],  # 42: 'right_index3',
                              [0, -1, 0],  # 43: 'right_middle1',
                              [0, -1, 0],  # 44: 'right_middle2',
                              [0, -1, 0],  # 45: 'right_middle3',
                              [0, -1, 0],  # 46: 'right_pinky1',
                              [0, -1, 0],  # 47: 'right_pinky2',
                              [0, -1, 0],  # 48: 'right_pinky3',
                              [0, -1, 0],  # 49: 'right_ring1',
                              [0, -1, 0],  # 50: 'right_ring2',
                              [0, -1, 0],  # 51: 'right_ring3',
                              [0, -1, 0],  # 52: 'right_thumb1',
                              [0, -1, 0],  # 53: 'right_thumb2',
                              [0, -1, 0]])  # 54: 'right_thumb3'

joints_num = len(n_raw_offsets)
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]

# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]


def process_file(positions, feet_thre, n_joints):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    print("source motion:", positions.shape)
    positions = positions[:, :n_joints]     # 将positions截取到指定的关节数量n_joints
    print("after cut:", positions.shape)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]     # 计算地面高度，即positions中最小的y坐标值
    positions[:, :, 1] -= floor_height      # 将所有位置的y坐标减去地面高度，使运动在地面上进行
    #     print(floor_height)

    # print(positions.shape)
    # plot_3d_motion(os.path.join(save_path, "positions_1.mp4"), kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''      # XZ平面原点对齐
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx        # 定义四个关节索引：右髋关节（r_hip）、左髋关节（l_hip）、右肩关节（sdr_r）和左肩关节（sdr_l）
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])      # 定义目标朝向向量为Z轴正方向
    root_quat_init = qbetween_np(forward_init, target)      # 计算初始根关节四元数root_quat_init，使初始朝向向量与目标朝向向量对齐
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)      # 使用root_quat_init对positions进行旋转

    # print(positions.shape)
    # plot_3d_motion(os.path.join(save_path, "positions_2.mp4"), kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*', color='y')
    plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axis('equal')
    # plt.savefig(os.path.join(save_path, 'trajectory_1.png'))

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]     # 2023.8.22
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)
        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        # np.isnan(quat_params).any()
        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)       # np.isnan(cont_6d_params).any()
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    # r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])
    plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axis('equal')
    # plt.savefig(os.path.join(save_path, 'trajectory_2.png'))

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # print(np.isnan(root_data).any(),
          # np.isnan(ric_data).any(),
          #   np.isnan(rot_data).any(),
          #   np.isnan(local_vel).any(),
          #   np.isnan(feet_l).any(),
          #   np.isnan(feet_r).any())

    data = root_data        # (len, 4)
    data = np.concatenate([data, ric_data[:-1]], axis=-1)       # ric_data[:-1] (len, 378)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)       # (len, 756)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)       # (len, 381)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)      # (len, 2), (len, 2)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

def recover_from_ric(data, joints_num):
    # 调用recover_root_rot_pos函数，从输入数据中恢复根节点的旋转四元数和位置信息
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # 从输入数据中提取关节位置信息
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    # 调整张量形状，使其最后一个维度为3
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # 将Y轴旋转信息添加到局部关节位置信息中
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # 将根节点的XZ位置信息添加到关节位置信息中
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # 将根节点位置信息与关节位置信息进行拼接
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    # 返回恢复后的关节位置信息
    return positions


def recover_root_rot_pos(data):
    # 提取旋转速度信息，即data的最后一个维度的第一个值
    rot_vel = data[..., 0]
    # 初始化一个与rot_vel形状相同的全零张量r_rot_ang，用于存储Y轴旋转角度
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # 从旋转速度中获取Y轴的旋转角度
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    # 计算累计和，以得到每个时间步的总旋转角度
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # 初始化一个四元数张量r_rot_quat，用于存储每个时间步的旋转四元数
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # 使用Y轴旋转角度计算旋转四元数的实部和虚部
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # 初始化一个位置张量r_pos，用于存储每个时间步的根节点位置信息
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    # 提取输入数据中的X和Z轴位置信息
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    # 将Y轴旋转信息添加到根节点位置信息中
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    # 计算累计和，以得到每个时间步的根节点总位置
    r_pos = torch.cumsum(r_pos, dim=-2)

    # 提取输入数据中的Y轴位置信息
    r_pos[..., 1] = data[..., 3]
    # 返回旋转四元数和根节点位置信息
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def fold_joints_to_feat(source_fold, target_fold, n_joints=55):
    if not os.path.exists(target_fold):
        os.makedirs(target_fold)
    error_files = []
    for file in os.listdir(source_fold):
        if os.path.exists(os.path.join(target_fold, file)):
            print('file: ', file, ' already exists')
            continue
        if file.endswith('.npy'):
            print('processing file: ', file)
            source_joints = np.load(os.path.join(source_fold, file))
            # data, _, _, _ = process_file(source_joints, feet_thre=0.02, n_joints=n_joints)
            try:
                data, _, _, _ = process_file(source_joints, feet_thre=0.02, n_joints=n_joints)
                if np.isnan(data).any() or np.isinf(data).any():
                    print('error file: ', file)
                    error_files.append(file)
                    continue
            except:
                print('error file: ', file)
                error_files.append(file)
                continue
            print("data.shape", data.shape)
            np.save(os.path.join(target_fold, file), data)
    print('error_files: ', error_files)     # error_files:  ['M000990.npy']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_fold', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_downsample/train/motion_joints")
    parser.add_argument('--target_fold', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_downsample/train/motion_joints_vecs_3")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    python process/motion_representation.py
    '''

    '''
    source_path = "/apdcephfs/private_yyyyyyyang/tmp/123456.npy"       # 123456, 2_scott_0_75_75
    save_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    source_joints = np.load(source_path)
    data, ground_positions, positions, l_velocity = process_file(source_joints, feet_thre=0.02)
    rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
    pdb.set_trace()
    print("data.shape: ", data.shape, 'rec_ric_data.shape', rec_ric_data.squeeze().numpy().shape)
    plot_3d_motion(os.path.join(save_path, "positions_3.mp4"), kinematic_chain, rec_ric_data.squeeze().numpy(), 'title', fps=20)
    np.save(os.path.join(save_path, "2_scott_0_75_75_rec_ric_data.npy"), rec_ric_data.squeeze().numpy())
    '''

    # save_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    # source_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/val/motion_joints"
    # target_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/val/motion_joints_vecs_2"      # feature dim 659
    # fold_joints_to_feat(source_fold, target_fold, n_joints=55)
    #
    # source_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/train/motion_joints"
    # target_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/train/motion_joints_vecs_2"
    # fold_joints_to_feat(source_fold, target_fold, n_joints=55)
    #
    # source_fold = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_2"
    # target_fold = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs_2"
    # fold_joints_to_feat(source_fold, target_fold, n_joints=55)

    args = get_args()
    source_fold = args.source_fold
    target_fold = args.target_fold
    # print(source_fold, target_fold)
    fold_joints_to_feat(source_fold, target_fold, n_joints=55)

    # save_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    # source_fold = "/apdcephfs/share_1290939/new_data/BEAT/my_downsample/val/motion_joints"
    # target_fold = "/apdcephfs/share_1290939/new_data/BEAT/my_downsample/val/motion_joints_vecs_3"      # feature dim 659
    # fold_joints_to_feat(source_fold, target_fold, n_joints=55)

