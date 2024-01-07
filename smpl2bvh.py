import pdb

import numpy as np
import pickle
import sys
sys.path.append('/apdcephfs/private_yyyyyyyang/code/')

def smpl2bvh(pos, bvh_path):
    from Motion.InverseKinematics import animation_from_positions
    from Motion import BVH
    # motion_path = f'/path/{npy_file}'
    # pos = np.load(motion_path)
    pos = pos.transpose(0, 3, 1, 2) # samples x joints x coord x frames ==> samples x frames x joints x coord
    # parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

    SMPL_JOINT_NAMES = [
    'Pelvis', # 0
    'L_Hip', # 1
    'R_Hip', # 2
    'Spine1', # 3
    'L_Knee', # 4
    'R_Knee', # 5
    'Spine2', # 6
    'L_Ankle', # 7
    'R_Ankle', # 8
    'Spine3', # 9
    'L_Foot', # 10
    'R_Foot', # 11
    'Neck', # 12
    'L_Collar', # 13
    'R_Collar', # 14
    'Head', # 15
    'L_Shoulder', # 16
    'R_Shoulder', # 17
    'L_Elbow', # 18
    'R_Elbow', # 19
    'L_Wrist', # 20
    'R_Wrist', # 21
    'L_Hand', # 22
    'R_Hand', # 23
    ]
    for i, p in enumerate(pos):
        print(f'starting anim no. {i}')
        anim, sorted_order, _ = animation_from_positions(p, parents)
        BVH.save(bvh_path.format(i), anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])


def main1():
    result_path = "/apdcephfs/private_yyyyyyyang/code/mdm/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/"
    source_result = result_path + "results.npy"
    source_result = np.load(source_result, allow_pickle=True)
    source_result_dict = source_result.item()       # 'motion', 'text', 'lengths', 'num_samples', 'num_repetitions'
    source_result_motion = source_result_dict['motion']     # (1, 22, 3, 196)
    save_path = result_path + "results.bvh"
    smpl2bvh(source_result_motion, save_path)


def main2():
    result_path = "/apdcephfs/private_yyyyyyyang/code/mdm/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/"
    source_result = result_path + "sample00_rep00_smpl_params.npy.pkl"
    save_path = result_path + "sample00_rep00_smpl_params_scaling.pkl"

    with open(source_result, "rb") as fp:
        data = pickle.load(fp)
        data_dict = {'smpl_poses': data['smpl_poses'],'smpl_trans': data['smpl_trans'],'smpl_scaling': [100, 1, 100]}
        with open(save_path +".pkl", 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)


def main3():
    sys.path.append('/apdcephfs/private_yyyyyyyang/code/fairmotion')
    from fairmotion.data import amass, bvh
    import os

    # prefix = f'./<data-dir-path-here>'  # eg: './AMASS/TotalCapture/s1'

    # list all the files in the directory
    # files = [f for f in os.listdir(prefix)]

    # prefix = f'/apdcephfs/private_yyyyyyyang/data/SSM/20160330_03333/'
    # files = ["ankles_stageii.npz"]

    prefix = f"/apdcephfs/private_yyyyyyyang/data/SMPLH/SSM_synced/20160330_03333/"
    files = ["ankles_poses.npz"]

    # bm_path = './<body-model-path-here>'  # eg: body_models/smplh/neutral/model.npz'
    # bm_path = "/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models/smplx/neutral/model.npz"
    bm_path = "/apdcephfs/private_yyyyyyyang/data/SMPLH/neutral/model.npz"
    motion = [bvh.save(amass.load(os.path.join(prefix, f), bm_path=bm_path, model_type="smplh"),
                       f"{os.path.join(prefix, f.split('.')[0]) + '.bvh'}") for f in files]


def amass_to_blender():
    source_npy = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/amass/000000.npy"
    motion = np.load(source_npy, allow_pickle=True)
    print(motion.shape)
    target_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    # np.savez(target_path + "000000.npz", global_rot=motion[:, :3], body_pose=motion[:, 3:66], jaw_pose=motion[:, 66:75],
    #          hand_pose=motion[:, 75:165], global_trans=motion[:, 165:168], body_shape=motion[:, 168: 178],
    #          foot_contact=motion[:, 178: 182], face_expr=motion[:, 182: 192], gender=motion[:, 192:193])     # femanle = 0，male =1，netural = 2，没有的统一为netural)
    np.savez(target_path + "000000.npz", poses=motion[:, :165], trans=motion[:, 165:168],
             betas=np.concatenate([motion[:, 168:178][0], np.zeros((6))], axis=0),
             # betas=motion[:, 168:178],
             gender='male', surface_model_type='smplx', mocap_frame_rate=30)


def process_amass():
    source_npz = "/apdcephfs/private_yyyyyyyang/data/SSM/20160330_03333/ankles_stageii.npz"
    target_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    motion = np.load(source_npz, allow_pickle=True)
    begin_trans = motion['trans'][0]
    norm_trans = motion['trans']-begin_trans
    np.savez(target_path + "ankles_stageii_process.npz",
             gender=motion['gender'], surface_model_type=motion['surface_model_type'], mocap_frame_rate=motion['mocap_frame_rate'],
             mocap_time_length=motion['mocap_time_length'], trans=norm_trans, poses=motion['poses'], betas=motion['betas'])


def beat_to_blender():
    source_npz = "/apdcephfs/private_yyyyyyyang/tmp/wayne_0_5_5(1).npz"
    target_path = "/apdcephfs/private_yyyyyyyang/tmp/"
    motion = np.load(source_npz, allow_pickle=True)
    len_motion = motion['poses'].shape[0]
    begin_trans = motion['trans'][0]
    norm_trans = motion['trans'] - begin_trans
    np.savez(target_path + "wayne_0_5_5(1)_process.npz",
             gender=motion['gender'], surface_model_type='smplx', mocap_frame_rate=motion['mocap_framerate'],
             poses=motion['poses'].reshape(len_motion, -1), betas=motion['betas'], trans=norm_trans)


if __name__ == '__main__':
    '''
    python smpl2bvh.py
    '''
    # main1()
    # main2()
    # main3()
    # amass_to_blender()
    # process_amass()

    beat_to_blender()
