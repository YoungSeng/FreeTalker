import pdb
import h5py
import numpy as np
import os
import subprocess
import copy
import torch


def read_data(h5file="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v0_val.h5",
                            statistics_path="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/", version="v0"):
    h5 = h5py.File(h5file, "r")
    len_h5 = len(h5.keys())
    mean = np.load(os.path.join(statistics_path, version + "_mean.npy"))
    std = np.load(os.path.join(statistics_path, version + "_std.npy"))
    dataset = [h5[str(i)]["dataset"][:][0] for i in range(len_h5)]
    audio = [h5[str(i)]["audio"][:] for i in range(len_h5)]
    text = [h5[str(i)]["text"][:] for i in range(len_h5)]
    motion = [(h5[str(i)]["motion"][:] - mean) / std for i in range(len_h5)]
    h5.close()
    print("Total clips:", len(motion))

    dataset_count = np.array([dataset.count(b'BEAT'), dataset.count(b'HUMANML3D')])

    return motion, audio, text, dataset_count


def pick_data(BEAT_text_path, HUMANML3D_text_path, n_sample_audio, n_sample_text):

    sample_audio_files = []
    sample_text_files = []
    with open(BEAT_text_path + "val.txt", 'r') as f:
        for v_i, line in enumerate(f.readlines()):
            print(line.strip())
            sample_audio_files.append(line.strip())
            v_i += 1
            if v_i == n_sample_audio:
                break
    with open(HUMANML3D_text_path + "val.txt", 'r') as f:
        for v_i, line in enumerate(f.readlines()[2:]):
            print(line.strip())
            sample_text_files.append(line.strip())
            v_i += 1
            if v_i == n_sample_text:
                break
    return sample_audio_files, sample_text_files


def read_data(sample_audio_files, sample_text_files, segment_length, BEAT_wav_feat, HUMANML3D_text_feat, save_path, GT_HUMANML3D_motion):

    sample_audio_feat = []
    sample_text_feat = []
    text = []

    for audio_file in sample_audio_files:
        speaker = audio_file.split("_")[0]
        audio_feat = np.load(os.path.join(BEAT_wav_feat, speaker, audio_file + ".npy"))
        audio_feat = audio_feat[:segment_length]
        sample_audio_feat.append(audio_feat)
    for text_file in sample_text_files:
        text_feat = np.load(os.path.join(HUMANML3D_text_feat, text_file + ".npz"), allow_pickle=True)['text_data'][0]
        text.append(text_feat['caption'])
        text_feat = text_feat['caption_clip']
        sample_text_feat.append(text_feat)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        subprocess.call(["cp", "-r", os.path.join(GT_HUMANML3D_motion, text_file + ".npz"), save_path])
    print(text)
    return sample_audio_feat, sample_text_feat


def sample_cond(model, sample_fn, cond, audio, text, seed, this_len=190):
    torch.manual_seed(123456)
    if 't2m' in cond:
        audio = np.zeros((1, segment_length, 1133))
    elif 'a2g' in cond:
        text = np.zeros((1, 512))
    model_kwargs_ = {'y': {}}

    model_kwargs_['y']['audio'] = torch.from_numpy(audio).to(device).float()[:, n_seed:this_len]
    model_kwargs_['y']['text'] = torch.from_numpy(text).to(device).float()
    if seed != None:
        model_kwargs_['y']['seed'] = seed
    model_kwargs_['y']['mask'] = torch.ones(1, this_len).bool().to(device)
    shape_ = (1, model.njoints, model.nfeats, this_len)

    sample = sample_fn(
        model,
        shape_,
        clip_denoised=False,
        model_kwargs=model_kwargs_,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,  # None, torch.randn(*shape_, device=mydevice)
        const_noise=False,
    )
    return sample

def smoothing_motion_transition(out_list_, sample):
    # smoothing motion transition
    if len(out_list_) > 0:
        last_poses = out_list_[-1][..., -n_seed:]  # (1, model.njoints, 1, args.n_seed)
        out_list_[-1] = out_list_[-1][..., :-n_seed]  # delete last n_seed frames
        # print('last_poses.shape', last_poses.shape)     # [1, 168, 1, 20]
        for j in range(n_seed):
            prev = last_poses[..., j]
            next = sample[..., j]
            sample[..., j] = prev * (n_seed - j) / (n_seed + 1) + next * (j + 1) / (n_seed + 1)
    out_list_.append(sample)
    return out_list_


def process(sample_audio_feat, sample_text_feat, segment_length, save_path, model_name, n_seed=10, version="v0"):
    from utils.model_util import create_model_and_diffusion
    from utils.model_util import load_model_wo_clip
    from utils.parser_util import train_args
    args = train_args()

    # text2motion

    text1 = sample_text_feat[0]
    # aux = np.load("/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/10/10_kieks_0_12_12.npz", allow_pickle=True)      # note: 30fps
    if 'v0' in version:
        aux = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/005204.npz", allow_pickle=True)
        seed1 = np.concatenate((aux['poses'].reshape(aux['poses'].shape[0], -1), aux['trans']), axis=1)[:n_seed]
    elif ('v1' in version):
        aux = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs/005204.npy")
        seed1 = aux[:n_seed]
    elif (('v2' in version) or ('v3' in version)) and n_seed != 0:
        aux = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs_2/005204.npy")
        seed1 = aux[:n_seed]

    mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/" + version.split('_')[0] + "_mean.npy")
    std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/"+ version.split('_')[0] +"_std.npy")
    if 'v0' in version or 'v1' in version or ('v2' in version and n_seed != 0) or ('v3' in version and n_seed != 0):
        seed1 = (seed1 - mean) / std
        seed1 = torch.from_numpy(seed1).to(device).float().unsqueeze(0).unsqueeze(0).permute(1, 3, 0, 2)

    model_path = os.path.join("/apdcephfs/private_yyyyyyyang/code/mdm/save/my_" + version + "/", model_name + ".pt")
    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, model_version=version, device=device)
    sample_fn = diffusion.p_sample_loop
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()

    if 'v0' in version or 'v1' in version or ('v2' in version and n_seed != 0) or ('v3' in version and n_seed != 0):
        out_list = []
        sample1 = sample_cond(model, sample_fn, 't2m', None, text1, seed1, this_len=120)
        out_list.append(sample1)
        audio2 = np.expand_dims(sample_audio_feat[0], axis=0)
        seed2 = out_list[-1][..., -n_seed:]
        sample2 = sample_cond(model, sample_fn, 'a2g', audio2, None, seed2, this_len=segment_length)
        out_list__ = copy.deepcopy(out_list)
        out_list = smoothing_motion_transition(out_list__, sample2)
        text2 = sample_text_feat[1]
        seed3 = out_list[-1][..., -n_seed:]
        sample3 = sample_cond(model, sample_fn, 't2m', None, text2, seed3, this_len=120)
        out_list__ = copy.deepcopy(out_list)
        out_list = smoothing_motion_transition(out_list__, sample3)
    elif 'v2' in version:
        out_list = []
        sample1 = sample_cond(model, sample_fn, 't2m', None, text1, None, this_len=120)
        out_list.append(sample1)
        audio2 = np.expand_dims(sample_audio_feat[0], axis=0)
        sample2 = sample_cond(model, sample_fn, 'a2g', audio2, None, None, this_len=160)
        out_list.append(sample2)
        text2 = sample_text_feat[1]
        sample3 = sample_cond(model, sample_fn, 't2m', None, text2, None, this_len=120)
        out_list.append(sample3)
    else:
        raise NotImplementedError

    out_list = [i.detach().data.cpu().numpy() for i in out_list]

    # if len(out_list) > 1:
    #     out_dir_vec_1 = np.vstack(out_list[:-1])
    #     sampled_seq_1 = out_dir_vec_1.squeeze(2).transpose(0, 2, 1).reshape(1, -1, 168)
    #     out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
    #     sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    # else:
    #     sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = np.concatenate(out_list, axis=-1).squeeze(2).transpose(0, 2, 1)      # (1, len, 168)
    sampled_seq = sampled_seq[:, n_seed:]

    out_poses = np.multiply(sampled_seq[0], std) + mean

    if "v0" in version:
        np.savez(os.path.join(save_path, "result.npz"), gender="neutral", surface_model_type='smplx', mocap_frame_rate=20,
                 poses=out_poses[:, :165], trans=out_poses[:, 165:168], betas=np.zeros(16))
    elif "v1" in version or "v2" in version or "v3" in version:
        import sys
        [sys.path.append(i) for i in ['./process']]
        from process.motion_representation import plot_3d_motion, kinematic_chain, recover_from_ric, joints_num
        rec_ric_data = recover_from_ric(torch.from_numpy(out_poses).unsqueeze(0).float(), joints_num)[0]
        print("rec_ric_data shape: ", rec_ric_data.shape)
        plot_3d_motion(os.path.join(save_path, "positions.mp4"), kinematic_chain, np.array(rec_ric_data), 'title', fps=20)

        np.save(os.path.join(save_path, "result.npy"), out_poses)

    print("Done!")

    # audio2motion


if __name__ == '__main__':
    '''
    python -m sample.sample_mdm --save_dir '' (--batch_size 256 --n_frames 190 --split_para 32)
    '''
    # motion, audio, text, dataset_count = read_data()

    BEAT_text_path = "/apdcephfs/share_1290939/new_data/BEAT/"
    HUMANML3D_text_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/"
    n_sample_audio = 1
    n_sample_text = 2

    sample_audio_files, sample_text_files = pick_data(BEAT_text_path, HUMANML3D_text_path, n_sample_audio, n_sample_text)
    print(sample_audio_files, sample_text_files)
    BEAT_wav_feat = "/apdcephfs/share_1290939/new_data/BEAT/BEAT_wav_feat/"
    HUMANML3D_text_feat = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/HUMANML3D_txt_feat/"
    version = 'v3_7'
    model_name = "model000800000"
    if 'v2' in version:
        segment_length = 160
    elif version == 'v3_6' or version == 'v3_7':
        segment_length = 180
    else:
        segment_length = 190
    save_path = os.path.join("/apdcephfs/private_yyyyyyyang/code/mdm/save", "my_" + version)
    save_path = os.path.join(save_path, model_name)
    print(save_path)
    GT_HUMANML3D_motion = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/"
    # "/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/"
    sample_audio_feat, sample_text_feat = read_data(sample_audio_files, sample_text_files, segment_length, BEAT_wav_feat, HUMANML3D_text_feat, save_path, GT_HUMANML3D_motion)

    if version == 'v0_0' or version == 'v0_1':
        n_seed = 10
    elif version == 'v0_2' or version == 'v0_3' or version == 'v0_4' or version == 'v0_5' or version == "v1_2" or version == "v1_3" or version == "v1_4":
        n_seed = 20
    elif version == "v0_2_1" or version == "v0_3_1":
        n_seed = 40
    elif version == 'v2_2' or version == 'v2_3' or version == 'v2_7' or version == 'v2_8' or version == 'v3_6' or version == 'v3_7':
        n_seed = 30
    else:
        raise ValueError("version not found!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    process(sample_audio_feat, sample_text_feat, segment_length, save_path, model_name, n_seed=n_seed, version=version)
