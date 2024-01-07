import numpy as np
import sys
[sys.path.append(i) for i in ['./process']]
from process.plot_script import plot_3d_motion as plot_3d_motion_1
from process.motion_representation import plot_3d_motion as plot_3d_motion_2
from process.motion_representation import kinematic_chain, recover_from_ric, joints_num, process_file
import torch
import os
import pdb
import matplotlib.pyplot as plt
import subprocess


n_joints = 55

def filter_line():
    import numpy as np
    import matplotlib.pyplot as plt

    # 示例数据
    filter = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                       0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                       0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0])

    num_points = len(filter)
    curve = np.random.random(num_points)  # 示例曲线，随机生成与filter相同数量的点

    # 平滑窗口大小
    window_size = 5

    # 对曲线进行平滑处理
    smoothed_curve = curve.copy()
    for i in range(len(filter)):
        if filter[i] == 1:
            start = max(0, i - window_size // 2)
            end = min(len(filter), i + window_size // 2 + 1)
            smoothed_curve[i] = np.mean(curve[start:end])

    # 创建一个新的图形
    plt.figure()

    # 绘制初始曲线
    plt.plot(curve, label='Original Curve', linestyle='dashed')

    # 绘制处理后的曲线
    plt.plot(smoothed_curve, label='Smoothed Curve')

    # 添加图例
    plt.legend()

    # 添加标题
    plt.title('Original vs Smoothed Curve')

    # 保存图形到文件
    plt.savefig('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/curve_comparison.png')

    # 显示图形
    # plt.show()


if __name__ == '__main__':
    '''
    python -m process.tmp_test
    '''

    '''
    source_file = "/apdcephfs/share_1290939/new_data/BEAT/my_downsample/val/motion_joints_vecs_3/8_catherine_0_82_82.npy"
    save_path = "/apdcephfs/private_yyyyyyyang/tmp"
    x = np.load(source_file)
    print(x.shape)      # (1019, 659)
    y = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs_2/000000.npy")

    mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_mean.npy")
    # std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_std.npy")

    x1 = x[:100]        # (100, 659)
    joints_num = 55
    rec_ric_data = recover_from_ric(torch.from_numpy(x1).unsqueeze(0).float(), joints_num)[0]
    print("rec_ric_data shape: ", rec_ric_data.shape)


    x2 = x[20:120]  # (100, 659)
    rec_ric_data2 = recover_from_ric(torch.from_numpy(x2).unsqueeze(0).float(), joints_num)[0]
    print("rec_ric_data shape: ", rec_ric_data2.shape)

    plot_3d_motion_2(os.path.join(save_path, "positions_my_1.mp4"), kinematic_chain, np.array(rec_ric_data), 'title', fps=20)
    # plot_3d_motion_2(os.path.join(save_path, "positions_my_2.mp4"), kinematic_chain, np.array(rec_ric_data2), 'title', fps=20)
    plot_3d_motion_2(os.path.join(save_path, "positions_my.mp4"), kinematic_chain, np.array(rec_ric_data[:50]), 'title',
                     fps=20, joints2=np.array(rec_ric_data[50:100]))

    '''


    source_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_2.npy"

    source_data = np.load(source_file)
    print(source_data.shape)

    joints_num = 55
    root_data = source_data[:, :4]
    ric_data = source_data[:, 4:4+(joints_num-1)*3].reshape(-1, joints_num-1, 3)
    rot_data = source_data[:, 4+(joints_num-1)*3:4+(joints_num-1)*3+(joints_num-1)*6].reshape(-1, joints_num-1, 6)
    local_vel = source_data[:, 4+(joints_num-1)*3+(joints_num-1)*6:4+(joints_num-1)*3+(joints_num-1)*6+joints_num*3].reshape(-1, joints_num, 3)
    feet_contact = source_data[:, 4+(joints_num-1)*3+(joints_num-1)*6+joints_num*3:]
    print("root_data shape: ", root_data.shape, "ric_data shape: ", ric_data.shape, "rot_data shape: ", rot_data.shape, "local_vel shape: ", local_vel.shape, "feet_contact shape: ", feet_contact.shape)


    '''

    l_velocity = root_data[:, 2:4]          # root linear velocity
    # root_data[:, 2:4] = 0
    # local_vel[:][:] = 0       # no influence

    feet_contact_sum = np.sum(feet_contact, axis=1)     # mean [0.9760123 , 0.96965134, 0.9764041 , 0.9687551 ] std [0.15373962, 0.17236596, 0.15252739, 0.17487194]
    threshold = 0.75
    # 根据条件修改root_data

    save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000"
    mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_mean.npy")
    std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_std.npy")
    output_data = np.concatenate((root_data, ric_data.reshape(-1, (joints_num-1)*3), rot_data.reshape(-1, (joints_num-1)*6), local_vel.reshape(-1, joints_num*3), feet_contact), axis=1)
    output_data = np.multiply(output_data, std) + mean
    rec_ric_data = recover_from_ric(torch.from_numpy(output_data).unsqueeze(0).float(), joints_num)[0]
    print("rec_ric_data shape: ", rec_ric_data.shape)

    filter = np.zeros(len(feet_contact_sum))
    for i in range(len(feet_contact_sum)):
        if feet_contact_sum[i] > threshold:
            filter[i] = 1

    print("filter: ", filter)
    root_joint = rec_ric_data[:, 0, :]       # [0, 2]
    print("root_joint shape: ", root_joint.shape)

    # smoothed_root_joint = root_joint.clone()
    # def gaussian_weights(window_size, sigma=1):
    #     # 计算高斯权重
    #     x = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    #     weights = torch.exp(-0.5 * (x / sigma) ** 2)
    #     return weights / weights.sum()
    #
    #
    # # 平滑窗口大小
    # window_size = 11
    # sigma = 20  # 增加高斯平滑的标准差以减小平滑效果
    #
    # # 计算高斯权重
    # weights = gaussian_weights(window_size, sigma)
    #
    # # 对曲线进行平滑处理

    # for i in range(len(filter)):
    #     if filter[i] == 1:
    #         # 计算滑动窗口的起始和结束索引
    #         start = max(0, i - window_size // 2)
    #         end = min(len(filter), i + window_size // 2 + 1)
    #
    #         # 计算滑动窗口内的加权平均值，并将其赋值给平滑后的张量
    #         window = root_joint[start:end]
    #         window_weights = weights[:len(window)] if start > 0 else weights[-len(window):]
    #         smoothed_root_joint[i] = torch.sum(window * window_weights[:, None], dim=0) / window_weights.sum()

    def ewma_varying_alpha(tensor, alpha, filter, window_size=1):
        smoothed_tensor = tensor.clone()
        for i in range(1, len(tensor)):
            if filter[i] == 1:      # or (any(filter[max(0, i - window_size):i]) and any(filter[i + 1:i + window_size + 1]))
                alpha_i = alpha[1]
            else:
                alpha_i = alpha[0]
            smoothed_tensor[i] = alpha_i * tensor[i] + (1 - alpha_i) * smoothed_tensor[i - 1]
        return smoothed_tensor


    # 对整个曲线应用EWMA平滑处理，根据滤波器值调整alpha参数
    window_size = 1  # 窗口大小，用于确定滤波器为0的点附近的点
    alpha = [0.1, 0]  # alpha[0]用于滤波器为0的点，alpha[1]用于滤波器为1的点
    # smoothed_root_joint = ewma_varying_alpha(root_joint, alpha, filter)
    # 计算原始曲线的差分（即斜率）
    root_joint_diff = torch.diff(root_joint, dim=0)

    # 对差分进行平滑处理
    smoothed_root_joint_diff = ewma_varying_alpha(root_joint_diff, alpha, filter, window_size)

    # 通过累积求和恢复平滑后的曲线
    smoothed_root_joint = torch.zeros_like(root_joint)
    smoothed_root_joint[0] = root_joint[0]
    smoothed_root_joint[1:] = torch.cumsum(smoothed_root_joint_diff, dim=0)

    # # 创建一个新的图形
    fig = plt.figure(figsize=(12, 6))
    # 创建一个3D子图来绘制原始数据
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Data')
    ax1.scatter(root_joint[:, 0], root_joint[:, 1], root_joint[:, 2], c='b', marker='o', label='Original Data')

    # 创建一个3D子图来绘制平滑后的数据
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Smoothed Data')
    ax2.scatter(smoothed_root_joint[:, 0], smoothed_root_joint[:, 1], smoothed_root_joint[:, 2], c='r', marker='o',
                label='Smoothed Data')
    # 保存图形到文件
    plt.savefig('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/tensor_comparison.png')

    # 创建一个子图来绘制原始数据
    # ax1 = fig.add_subplot(121)
    # ax1.set_title('Original Data')
    # ax1.scatter(root_joint[:, 0], root_joint[:, 1], c='b', marker='o', label='Original Data')
    #
    # # 创建一个子图来绘制平滑后的数据
    # ax2 = fig.add_subplot(122)
    # ax2.set_title('Smoothed Data')
    # ax2.scatter(smoothed_root_joint[:, 0], smoothed_root_joint[:, 1], c='r', marker='o', label='Smoothed Data')
    # plt.savefig('/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/tensor_comparison_2.png')

    # smoothed_root_joint = torch.cat((smoothed_root_joint[:, :1], rec_ric_data[:, 0, 1:2], smoothed_root_joint[:, 1:]), dim=1)
    # root_joint = torch.cat((root_joint[:, :1], rec_ric_data[:, 0, 1:2], root_joint[:, 1:]), dim=1)
    # delta = smoothed_root_joint - root_joint
    # delta = delta.unsqueeze(1)
    # delta = delta.repeat(1, joints_num, 1)
    # rec_ric_data[:, :, :] = rec_ric_data - delta

    delta = smoothed_root_joint - root_joint
    pdb.set_trace()
    delta = delta.unsqueeze(1)
    delta = delta.repeat(1, joints_num, 1)
    print(delta.shape, rec_ric_data.shape, smoothed_root_joint.shape, root_joint.shape)
    # torch.Size([125, 55, 3]) torch.Size([125, 55, 3]) torch.Size([125, 3]) torch.Size([125, 3])
    rec_ric_data = rec_ric_data + delta

    # rec_ric_data[:, 0] = smoothed_root_joint
    
    '''



    save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000"
    mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_mean.npy")
    std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_std.npy")
    output_data = np.concatenate((root_data, ric_data.reshape(-1, (joints_num-1)*3), rot_data.reshape(-1, (joints_num-1)*6), local_vel.reshape(-1, joints_num*3), feet_contact), axis=1)
    output_data = np.multiply(output_data, std) + mean
    rec_ric_data = recover_from_ric(torch.from_numpy(output_data).unsqueeze(0).float(), joints_num)[0]
    print("rec_ric_data shape: ", rec_ric_data.shape)


    def limit_global_motion(rec_ric_data, max_displacement=1.0, n_frames=20):
        # 计算前20帧的漂移
        drift = np.diff(rec_ric_data[:n_frames], axis=0)

        # 计算每帧的漂移大小
        drift_magnitude = np.linalg.norm(drift, axis=(1, 2))

        # 找到超过最大位移的帧
        exceed_max_displacement = drift_magnitude > max_displacement

        # 对超过最大位移的帧应用平滑滤波器
        smoothed_drift = np.where(
            exceed_max_displacement[:, np.newaxis, np.newaxis],
            drift * (max_displacement / drift_magnitude)[:, np.newaxis, np.newaxis],
            drift
        )

        # 更新前20帧的位置
        rec_ric_data[1:n_frames] = rec_ric_data[0] + np.cumsum(smoothed_drift, axis=0)

        # 计算原始漂移和平滑漂移之间的差值
        drift_diff = np.sum(drift - smoothed_drift, axis=0)

        # 更新后续帧的位置
        rec_ric_data[n_frames:] -= drift_diff

        return rec_ric_data


    def limit_global_motion_end(rec_ric_data, max_displacement=1.0, n_frames=20):
        # 获取数据的长度
        data_len = rec_ric_data.shape[0]

        # 计算后20帧的漂移
        drift = np.diff(rec_ric_data[-n_frames:], axis=0)

        # 计算每帧的漂移大小
        drift_magnitude = np.linalg.norm(drift, axis=(1, 2))

        # print("drift_magnitude: ", drift_magnitude)

        # 找到超过最大位移的帧
        exceed_max_displacement = drift_magnitude > max_displacement

        # 对超过最大位移的帧应用平滑滤波器
        smoothed_drift = np.where(
            exceed_max_displacement[:, np.newaxis, np.newaxis],
            drift * (max_displacement / drift_magnitude)[:, np.newaxis, np.newaxis],
            drift
        )

        # 更新后20帧的位置
        rec_ric_data[-n_frames+1:] = rec_ric_data[-n_frames] + np.cumsum(smoothed_drift, axis=0)

        return rec_ric_data

    # def smooth_motion(rec_ric_data, window_length=9, polyorder=2):
    #     from scipy.signal import savgol_filter
    #
    #     """
    #     平滑 rec_ric_data 中后 20 帧的整体 trans 运动。
    #
    #     参数:
    #     rec_ric_data (numpy array): 维度为 (len, 55, 3) 的数组，表示 55 个关节的三维坐标。
    #     window_length (int): 滤波器窗口长度。必须为正奇数。默认值为 5。
    #     polyorder (int): 滤波器多项式的阶数。默认值为 2。
    #
    #     返回:
    #     numpy array: 平滑后的 rec_ric_data。
    #     """
    #     smoothed_data = np.array(rec_ric_data.clone())
    #     last_20_frames = smoothed_data[-20:]
    #
    #     for joint in range(last_20_frames.shape[1]):
    #         for coord in range(last_20_frames.shape[2]):
    #             last_20_frames[:, joint, coord] = savgol_filter(last_20_frames[:, joint, coord], window_length,
    #                                                             polyorder)
    #
    #     smoothed_data[-20:] = last_20_frames
    #     return smoothed_data

    # def smooth_motion_ewma(rec_ric_data, alpha=0.01):
    #     import pandas as pd
    #
    #     """
    #     使用 EWMA 平滑 rec_ric_data 中后 20 帧的整体 trans 运动。
    #
    #     参数:
    #     rec_ric_data (numpy array): 维度为 (len, 55, 3) 的数组，表示 55 个关节的三维坐标。
    #     alpha (float): EWMA 的平滑系数，范围为 (0, 1)。较小的值表示对过去数据的更强平滑。默认值为 0.2。
    #
    #     返回:
    #     numpy array: 平滑后的 rec_ric_data。
    #     """
    #     smoothed_data = np.array(rec_ric_data.clone())
    #     last_20_frames = smoothed_data[-20:]
    #
    #     for joint in range(last_20_frames.shape[1]):
    #         for coord in range(last_20_frames.shape[2]):
    #             last_20_frames[:, joint, coord] = pd.Series(last_20_frames[:, joint, coord]).ewm(
    #                 alpha=alpha).mean().values
    #
    #     smoothed_data[-20:] = last_20_frames
    #     return smoothed_data

    # def smooth_global_motion(rec_ric_data, window_size=20):
    #     # 获取关节数据的形状
    #     num_frames, num_joints, _ = rec_ric_data.shape
    #
    #     # 创建一个新的数组来存储处理后的关节数据
    #     smoothed_data = np.copy(rec_ric_data.numpy())
    #
    #     # 计算后20帧的起始帧
    #     start_frame = max(0, num_frames - window_size)
    #
    #     # 对于每个关节，应用移动平均滤波器
    #     for joint in range(num_joints):
    #         # 计算关节在后20帧的平均位置
    #         avg_position = np.mean(rec_ric_data.numpy()[start_frame:, joint, :], axis=0)
    #
    #         # 将后20帧的关节位置设置为平均位置
    #         smoothed_data[start_frame:, joint, :] = avg_position
    #
    #     return smoothed_data


    max_displacement = 0.05
    n_frames = 20
    limited_rec_ric_data = limit_global_motion(rec_ric_data, max_displacement, n_frames)
    smoothed_data = limit_global_motion_end(limited_rec_ric_data, max_displacement, n_frames)

    new_array = torch.cat((smoothed_data[0].unsqueeze(0), smoothed_data), dim=0)

    data, _, _, _ = process_file(np.array(new_array), feet_thre=0.02, n_joints=n_joints)
    # print("data.shape: ", data.shape)
    # 复制第一维到最开始

    np.save(os.path.join(save_path, "positions_real_2_.npy"), data)

    # plot_3d_motion_1(os.path.join(save_path, "positions_real_2_debug.mp4"), kinematic_chain,
    #                  np.array(smoothed_data),
    #                  'title',
    #                  fps=20, dataset='humanml', vis_mode='gt')

    import matplotlib.pyplot as plt

    # 创建一个简单的图形
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 3, 4, 5]
    plt.plot(x, y)

    # 设置带换行的标题
    plt.title(r"Line Plot Example" + "\n" + r"with a Two-Line Title")

    # 显示图形
    plt.savefig(os.path.join(save_path, "test.png"))

    # pdb.set_trace()


