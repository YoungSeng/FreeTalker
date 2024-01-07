import pdb
import torch
import numpy as np
import glob
import os
import subprocess
import argparse


def element_by_element():
    x = torch.tensor([1000, 2000, 3000]).cuda()
    y = torch.tensor([400, 500, 600]).cuda()

    return x * y, torch.mul(x, y)


def mdm_to_recover(HumanML3D_path, save_path):
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    mdm_files = sorted(glob.glob(HumanML3D_path + "/*.npy"))
    for v_i, source_npy in enumerate(mdm_files):
        name = os.path.split(source_npy)[1][:-4]
        print(f"Processing {v_i + 1}/{len(mdm_files)}: {name}")
        save_path_ = os.path.join(save_path, name + ".npy")
        source_file = np.load(source_npy)       # (len, 263)
        source_file = np.expand_dims(source_file, axis=0)      # (1, len, 263)
        source_file = np.expand_dims(source_file, axis=0)      # (1, 1, len, 263)
        source_file = torch.from_numpy(source_file).float()    # (1, 1, len, 263)
        sample = recover_from_ric(source_file, 22)  # [1, 1, 196, 22, 3]
        sample = np.array(sample[0][0])       # (196, 22, 3)
        np.save(save_path_, sample)


def recover_to_smplx(recover_path, smplx_path):

    import sys
    # sys.path.append('/apdcephfs/private_yyyyyyyang/code/human_body_prior')
    sys.path.append('/ceph/hdd/yangsc21/Python/human_body_prior')

    print(sys.path)

    from tutorials.mdm_motion2smpl import convert_mdm_mp4_to_amass_npz

    print('recover_to_smplx')

    mdm_files = sorted(glob.glob(recover_path + "/*.npy"))
    finished_files = sorted(glob.glob(smplx_path + "/*.npz"))
    finished_files = [item.replace('vposer_smplx', 'new_joint_vecs_recover').replace('npz', 'npy') for item in finished_files]
    mdm_files = sorted(list(set(mdm_files) - set(finished_files)))
    n_mdm_files = len(mdm_files)
    # mdm_files = mdm_files[int(n_mdm_files * 0 / 6)+1:int(n_mdm_files * 1 / 6)]
    for v_i, source_npy in enumerate(mdm_files):
        name = os.path.split(source_npy)[1][:-4]
        print(f"Processing {v_i + 1}/{len(mdm_files)}: {name}", source_npy)
        target_path = smplx_path + name + ".npz"

        if os.path.exists(target_path):
            continue

        element_by_element()
        try:
            source_npy = np.load(source_npy)
        except:
            print("Error: ", source_npy)
            continue
        # subprocess.run(['python', '-m', 'mdm_motion2smpl', '--input', source_npy, '--output', target_path])

        print('convert_mdm_mp4_to_amass_npz')

        convert_mdm_mp4_to_amass_npz(skeleton_movie_fname=source_npy,
                                     out_fname=target_path,
                                     surface_model_type='smplx',
                                     gender='neutral',
                                     batch_size=128,
                                     save_render=False,
                                     verbosity=0)

        motion = np.load(target_path, allow_pickle=True)
        begin_trans = motion['trans'][0]
        norm_trans = motion['trans'] - begin_trans
        gender = motion['gender']
        surface_model_type = motion['surface_model_type']
        mocap_frame_rate = motion['mocap_frame_rate']
        poses = motion['poses']
        betas = motion['betas']

        np.savez(target_path,
                 gender=gender, surface_model_type=surface_model_type,
                 mocap_frame_rate=mocap_frame_rate,
                 trans=norm_trans, poses=poses,
                 betas=betas)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--HumanML3D_path', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/new_joint_vecs/")
    parser.add_argument('--save_path', type=str,
                        default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/new_joint_vecs_recover/")
    parser.add_argument('--smplx_path', type=str,
                        default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
     pip install colour trimesh
python -m process.mdm_to_smplx --HumanML3D_path "/ceph/hdd/yangsc21/Python/mdm/HumanML3D/HumanML3D/new_joint_vecs/" --save_path "/ceph/hdd/yangsc21/Python/mdm/HumanML3D/HumanML3D/new_joint_vecs_recover/" --smplx_path "/ceph/hdd/yangsc21/Python/mdm/HumanML3D/HumanML3D/vposer_smplx/"
    '''

    args = get_args()
    HumanML3D_path = args.HumanML3D_path
    save_path = args.save_path
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    # mdm_to_recover(HumanML3D_path, save_path)

    # cd /apdcephfs/private_yyyyyyyang/code/human_body_prior/tutorials/
    # conda activate human_body_prior, root位置报错不影响代码运行，只影响可视化；2080ti上才能运行，不然需要安装对应cuda的torch
    # python "/apdcephfs/private_yyyyyyyang/code/mdm/process/mdm_to_smplx.py"
    smplx_path = args.smplx_path
    if os.path.exists(smplx_path) is False:
        os.makedirs(smplx_path)
    recover_to_smplx(save_path, smplx_path)
