import os
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4, joints2=None):
#     matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        # ax.set_xlim3d([0, radius / 2])     # -radius / 2
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([0, radius])
        padding = 1  # 调整填充值以增加或减少空白区域
        ax.set_xlim3d([data[..., 0].min() - padding, data[..., 0].max() + padding])
        ax.set_ylim3d([data[..., 1].min() - padding, data[..., 1].max() + padding])
        ax.set_zlim3d([data[..., 2].min() - padding, data[..., 2].max() + padding])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred', 'darkred',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    # 从关节数据中减去了height_offset，trajec，以及关节数据本身的某些部分。这可能导致了关节位置的变化，从而产生了飘动的感觉。
    # height_offset = MINS[1]
    # data[:, :, 1] -= height_offset
    ankle_height = min(data[:, 7, 1].min(), data[:, 8, 1].min())
    data[:, :, 1] -= ankle_height

    # trajec = data[:, 0, [0, 2]]
    trajec = np.zeros_like(data[:, 0, [0, 2]])

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints2), -1, 3)
        data2[:, :, 1] -= ankle_height
        trajec2 = np.zeros_like(data2[:, 0, [0, 2]])


    def update(index):
        ax.lines = []
        ax.collections = []
        # ax.view_init(elev=120, azim=-90)
        ax.view_init(elev=30, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            if joints2 is not None:
                ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])


    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

if __name__ == '__main__':
    '''
    python process/animation.py
    '''
    kinematic_chain = [[0, 2, 5, 8, 11],  # 右侧下肢
                            [0, 1, 4, 7, 10], # 左侧下肢
                            [0, 3, 6, 9, 12, 15], # 脊柱和头部
                            [9, 14, 17, 19, 21], # 右侧上肢
                            [9, 13, 16, 18, 20], # 左侧上肢
                            [15, 22], # 颌部
                            [15, 23], # 左眼
                            [15, 24], # 右眼
                            [20, 25, 26, 27], # 左手食指
                            [20, 28, 29, 30], # 左手中指
                            [20, 31, 32, 33], # 左手小指
                            [20, 34, 35, 36], # 左手无名指
                            [20, 37, 38, 39], # 左手拇指
                            [21, 40, 41, 42], # 右手食指
                            [21, 43, 44, 45], # 右手中指
                            [21, 46, 47, 48], # 右手小指
                            [21, 49, 50, 51], # 右手无名指
                            [21, 52, 53, 54], # 右手拇指
                            ]
    source_path = "/apdcephfs/private_yyyyyyyang/tmp/123456.npy"
    save_path = "/apdcephfs/private_yyyyyyyang/tmp/123456.mp4"
    data = np.load(source_path)
    plot_3d_motion(save_path, kinematic_chain, data, title="None", fps=20, radius=4)
