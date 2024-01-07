import pdb

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
import os
import argparse


def load_npz(npz_path):
    data = np.load(npz_path)
    poses = data['poses']
    trans = data['trans']
    gender = data['gender']
    surface_model_type = data['surface_model_type']
    mocap_frame_rate = data['mocap_frame_rate']
    betas = data['betas']

    return poses, trans, gender, surface_model_type, mocap_frame_rate, betas


def smplx_model_init(model_path, batch_size=1):
    model = smplx.create(model_path, model_type='smplx', gender='neutral', ext='npz', num_pca_comps=12,
                         create_global_orient=True, create_body_pose=True, create_betas=True,
                         create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                         create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                         use_face_contour=False, batch_size=batch_size)        # 20230820
    model.use_pca = False
    return model


def smplx_frame_by_frame(poses, trans, model, result_vertices=False):
    '''
    :param poses: (len, 168)
    :param trans: (len, 3)
    :param model:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_params = torch.tensor(poses, dtype=torch.float32, device=device).unsqueeze(1)
    transl_params = torch.tensor(trans, dtype=torch.float32, device=device).unsqueeze(1)
    model.to(device)
    model.eval()
    result = []
    if result_vertices:
        vertices = []
    for frame in range(pose_params.shape[0]):
        output = model(body_pose=pose_params[frame, :, 3:66], global_orient=pose_params[frame, :, :3], transl=transl_params[frame],
                       jaw_pose=pose_params[frame, :, 66:69], leye_pose=pose_params[frame, :, 69:72],
                       reye_pose=pose_params[frame, :, 72:75], left_hand_pose=pose_params[frame, :, 75:120],
                       right_hand_pose=pose_params[frame, :, 120:165])
        result.append(output.joints.detach().cpu().numpy().squeeze())
        if result_vertices:
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
    if result_vertices:
        return np.array(result), np.array(vertices)
    else:
        return np.array(result)



def smplx_frame_by_frame_mdm(input, model):
    '''
    :param input: [bs, 168, 1, len] -> (len, 168)
    :param model:
    '''
    # input = input.permute(0, 3, 2, 1).squeeze(2)
    input = input.permute(0, 3, 2, 1)
    output_files = []
    for file in input:
        result = []
        for frame in range(file.shape[0]):

            output = model(global_orient=file[frame, :, :3], body_pose=file[frame, :, 3:66],
                           jaw_pose=file[frame, :, 66:69], leye_pose=file[frame, :, 69:72],
                            reye_pose=file[frame, :, 72:75], left_hand_pose=file[frame, :, 75:120],
                            right_hand_pose=file[frame, :, 120:165], transl=file[frame, :, 165:168])
            result.append(output.joints.detach().squeeze())

        # output = model(global_orient=file[..., :3], body_pose=file[..., 3:66],
        #                                   jaw_pose=file[..., 66:69], leye_pose=file[..., 69:72],
        #                                    reye_pose=file[..., 72:75], left_hand_pose=file[..., 75:120],
        #                                    right_hand_pose=file[..., 120:165], transl=file[..., 165:168])
        # output_files.append(output.joints.detach().squeeze())

        output_files.append(torch.stack(result))
    # return torch.stack(output_files).permute(0, 2, 3, 1)
    return torch.stack(output_files)


def smplx_frame_by_frame_mdm_2(input, model, split_para=1):
    '''
    :param input: [bs, 168, 1, len] -> (len, 168)
    :param model:
    '''
    file = input.permute(0, 3, 2, 1).squeeze()     # (bs, len, 168)
    bs, len, dim = file.shape
    file = file.reshape(bs * len, dim)       # (bs*len, 168)
    smplx_joints = 127
    smplx_joints_dim = 3
    output_files = []

    for i in range(split_para):
        begin = (bs*len*i)//split_para
        end = (bs*len*(i+1))//split_para
        output = model(global_orient=file[begin:end, :3], body_pose=file[begin:end, 3:66],
                      jaw_pose=file[begin:end, 66:69], leye_pose=file[begin:end, 69:72],
                       reye_pose=file[begin:end, 72:75], left_hand_pose=file[begin:end, 75:120],
                       right_hand_pose=file[begin:end, 120:165], transl=file[begin:end, 165:168])
        output_files.append(output.joints.detach().squeeze())

    return torch.cat(output_files, dim=0).reshape(bs, len, smplx_joints, smplx_joints_dim).permute(0, 2, 3, 1)


def extract_smplx_features(npz_path, model_path):

    poses, trans, gender, surface_model_type, mocap_frame_rate, betas = load_npz(npz_path)

    model = smplx.create(model_path, model_type='smplx', gender='neutral', ext='npz', num_pca_comps=12,
                         create_global_orient=True, create_body_pose=True, create_betas=True,
                         create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                         create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                         use_face_contour=False, batch_size=116)

    # num_betas?    num_expression_coeffs??
    # num_pca_comps 6 batch_size 1
    print(model)

    device = torch.device('cpu')
    batch_size = poses.shape[0]

    pose_params = torch.tensor(poses, dtype=torch.float32, device=device)
    transl_params = torch.tensor(trans, dtype=torch.float32, device=device)

    output = model(body_pose=pose_params[:, 3:66], global_orient=pose_params[:, :3], transl=transl_params)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    return vertices, joints, model.faces



def create_and_save_animation(joints_data, vertices_data, faces, output_file, plot_joints=True, plot_vertices=True, fps=20, dpi=100):
    interval = 1000 / fps  # 将FPS转换为毫秒间隔
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    def update(frame, ax):
        print(frame)
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if plot_vertices:
            vertices = vertices_data[frame]
            mesh = Poly3DCollection(vertices[faces], alpha=0.1)
            face_color = (1.0, 1.0, 0.9)
            edge_color = (0, 0, 0)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)

        if plot_joints:
            joints = joints_data[frame]
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)

        # ax.view_init(elev=90+20, azim=-90)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, fargs=(ax,), frames=len(joints_data), interval=interval, repeat=True)

    # 保存动画为MP4格式
    ani.save(output_file, writer='ffmpeg', dpi=dpi, bitrate=-1, extra_args=['-vcodec', 'libx264'])


def smplx_to_joints_folds(source_fold, target_fold, smplx_model_path, device, HUMANML3D_items=None):
    if not os.path.exists(target_fold):
        os.makedirs(target_fold)
    smplx_model = smplx_model_init(smplx_model_path, batch_size=1)
    smplx_model = smplx_model.eval().to(device)
    error_files = []
    for npz_file in os.listdir(source_fold):
        if npz_file.endswith('.npz'):
            if HUMANML3D_items is not None:
                if npz_file[:-4] not in HUMANML3D_items:
                    print('Skipping', npz_file[:-4])
                    continue
            npz_path = os.path.join(source_fold, npz_file)
            try:
                poses, trans, _, _, mocap_frame_rate, _ = load_npz(npz_path)
            except:
                error_files.append(npz_file)
                continue
            # assert mocap_frame_rate == 20
            poses = np.expand_dims(poses, axis=0)
            trans = np.expand_dims(trans, axis=0)
            input_file = np.expand_dims(np.concatenate((poses, trans), axis=-1), axis=0).transpose(0, 3, 1, 2)
            print('Processing', npz_file, 'input shape:', input_file.shape)
            input_file = torch.tensor(input_file, dtype=torch.float32).to(device)
            joints = smplx_frame_by_frame_mdm(input_file, smplx_model).cpu().numpy()
            print('Joints shape =', joints[0].shape)
            np.save(os.path.join(target_fold, npz_file[:-4] + '.npy'), joints[0])
    print('Error files:', error_files)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models")
    parser.add_argument('--source_fold', type=str,
                        default="/apdcephfs/share_1290939/new_data/BEAT/my_downsample/train/motion")
    parser.add_argument('--target_fold', type=str,
                        default="/apdcephfs/share_1290939/new_data/BEAT/my_downsample/train/motion_joints")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from motion_feat import extract_motion_features, reconstruct_joints_from_features
    '''
    python process/process_SMPLX.py
    SMPLX(
      Gender: NEUTRAL
      Number of joints: 55
      Betas: 10
      Number of PCA components: 6
      Flat hand mean: False
      Number of Expression Coefficients: 10
      (vertex_joint_selector): VertexJointSelector()
    )
    Vertices shape = (10475, 3)
    Joints shape = (127, 3)
    '''

    '''
    
    # npz_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/000000.npz"
    npz_path = "/apdcephfs/share_1290939/new_data/BEAT/downsample/val/motion/2_scott_0_75_75.npz"
    model_path = "/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models"

    # vertices, joints, faces = extract_smplx_features(npz_path, model_path)

    model = smplx_model_init(model_path, batch_size=1)
    poses, trans, gender, surface_model_type, mocap_frame_rate, betas = load_npz(npz_path)

    # joints, vertices = smplx_frame_by_frame(poses, trans, model, result_vertices=True)
    # print('Vertices shape =', vertices.shape)
    # print('Joints shape =', joints.shape)

    # poses = torch.tensor(poses, dtype=torch.float32).unsqueeze(0)
    # trans = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
    # input_file = torch.cat((poses, trans), axis=-1).unsqueeze(0).permute(0, 3, 1, 2)
    poses = np.expand_dims(poses, axis=0)
    trans = np.expand_dims(trans, axis=0)
    input_file = np.expand_dims(np.concatenate((poses, trans), axis=-1), axis=0).transpose(0, 3, 1, 2)
    print(input_file.shape)
    input_file = torch.tensor(input_file, dtype=torch.float32)

    # input_file = torch.rangedn(3, 168, 1, 190)
    joints = smplx_frame_by_frame_mdm(input_file, model)
    # joints = smplx_frame_by_frame_mdmm_2(input_file, model, split_para=2)
    print('Joints shape =', joints.shape)

    # motion_feat = extract_motion_features(torch.from_numpy(joints))
    # joints = reconstruct_joints_from_features(motion_feat)

    # 调用函数创建动画并保存为MP4格式
    # create_and_save_animation(joints, vertices, model.faces, 'animation.mp4')
    # np.save("/apdcephfs/private_yyyyyyyang/tmp/123456.npy", joints[0])
    np.save("/apdcephfs/private_yyyyyyyang/tmp/2_scott_0_75_75.npy", joints[0])
    create_and_save_animation(joints[0], None, None, 'animation_3.mp4', plot_vertices=False)
    '''

    args = get_args()
    model_path = args.model_path

    # source_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/val/motion"
    # target_fold = "/apdcephfs/share_1290939/new_data/BEAT/downsample/val/motion_joints"
    # smplx_to_joints_folds(source_fold, target_fold, model_path, device='cuda:0')


    # HUMANML3D_items = []
    # HUMANML3D_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/v2_train_val.txt"
    # with open(HUMANML3D_save_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         HUMANML3D_items.append(line.strip())

    # source_fold = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx"
    # target_fold = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_2"
    # smplx_to_joints_folds(source_fold, target_fold, model_path, device='cuda:0', HUMANML3D_items=HUMANML3D_items)       # Error files: ['M000454.npz', 'M000787.npz', '007870.npz', 'M000483.npz']




    source_fold = args.source_fold
    target_fold = args.target_fold
    smplx_to_joints_folds(source_fold, target_fold, model_path, device='cuda:0')
    # error_files: ['009707.npy', '011059.npy']

    # source_fold = "/apdcephfs/share_1290939/new_data/BEAT/my_downsample/val/motion"
    # target_fold = "/apdcephfs/share_1290939/new_data/BEAT/my_downsample/val/motion_joints"
    # smplx_to_joints_folds(source_fold, target_fold, model_path, device='cuda:0')
