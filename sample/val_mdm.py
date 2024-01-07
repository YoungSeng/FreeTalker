# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from train.training_loop import TrainLoop
from data_loaders.h5_data_loader import MotionDataset, t2m_collate, RandomSampler
from torch.utils.data import DataLoader
from utils.model_util import create_model_and_diffusion
import torch
import numpy as np
from tqdm import tqdm
import pdb
import codecs as cs
from utils.model_util import load_model_wo_clip
from utils.parser_util import train_args
args = train_args()
import math
import sys
[sys.path.append(i) for i in ['./process']]
from process.plot_script import plot_3d_motion as plot_3d_motion_1
# from process.motion_representation import plot_3d_motion as plot_3d_motion_2
from process.motion_representation import kinematic_chain, recover_from_ric, joints_num
from process.merge_mp4_audio import add_audio_to_video_pydub


def round_up_to_nearest_ten(number):
    return math.ceil(number / 10) * 10


def main():

    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    print("creating data loader...")
    dataset = MotionDataset(h5file="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v3_val.h5",
                            statistics_path="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/",
                            version='v3')

    data = DataLoader(dataset, num_workers=1,
                              batch_size=1,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=t2m_collate,
                      shuffle=False)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    print("Valuating...")

    all_captions_path = "/apdcephfs/private_yyyyyyyang/code/mdm/dataset/HumanML3D_/texts"
    all_captions = os.listdir(all_captions_path)
    handshake_size = 0
    source_BEAT_path = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"

    save_GT = False

    if save_GT:
        GT_save_path = os.path.join(save_path, "GT")
        if not os.path.exists(GT_save_path):
            os.makedirs(GT_save_path)

    generated_save_path = os.path.join(save_path, "generated_123456789")
    if not os.path.exists(generated_save_path):
        os.makedirs(generated_save_path)

    sample_fn = diffusion.p_sample_loop
    with torch.no_grad():

        for motion, cond in tqdm(data):

            motion = motion.to(device)      # [1, 659, 1, 180]
            cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in
                         cond['y'].items()}  # ['mask', 'lengths', 'text', 'tokens']

            name = cond['y']['from_data'][0].decode('utf-8')
            print(name)

            if os.path.exists(os.path.join(generated_save_path, name + ".npy")):
                print("Already generated:", name)
                continue

            Audio2Gesture = False
            if name + '.txt' not in all_captions:
                captions = "Audio2Gesture"
                Audio2Gesture = True
                # continue        # debug
            else:
                with cs.open(os.path.join(all_captions_path, name + '.txt')) as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        captions = line_split[0]

            length = cond['y']['lengths'].item()
            cond['y']['lengths'] = torch.tensor(round_up_to_nearest_ten(length)).unsqueeze(0).to(device)      # local attention channel
            # cond['y']['lengths'] = torch.tensor(180).unsqueeze(0).to(device)

            step_sizes = np.zeros(len(cond['y']['lengths']), dtype=int)
            for ii, len_i in enumerate(cond['y']['lengths']):
                if ii == 0:
                    step_sizes[ii] = len_i
                    # continue
                step_sizes[ii] = step_sizes[ii - 1] + len_i - handshake_size

            if save_GT:
                motion = motion.squeeze(2).permute(0, 2, 1).cpu().numpy()
                motion = np.multiply(motion, std) + mean
                rec_ric_data = recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), joints_num)[0]     # [1, 180, 55, 3]
                rec_ric_data = rec_ric_data.cpu().numpy()[0][:length]
                # print(rec_ric_data.shape)
                video_path = os.path.join(GT_save_path, "{}.mp4".format(name))

                plot_3d_motion_1(video_path,
                    kinematic_chain, np.array(rec_ric_data), captions,
                    fps=20, dataset='humanml', vis_mode='unfold_arb_len', handshake_size=0,
                    blend_size=0, step_sizes=step_sizes, lengths=cond['y']['lengths'])
                if Audio2Gesture:
                    video_withaudio_path = os.path.join(GT_save_path, "{}_with_audio.mp4".format(name))
                    add_audio_to_video_pydub(video_file=video_path, audio_tags=[[name.split('_')[0], name, 0, 180]],
                                             source_BEAT_path=source_BEAT_path,
                                             output_video_file=video_withaudio_path, merge_segments=[[0, 180]])
                np.save(os.path.join(GT_save_path, name + ".npy"), rec_ric_data)


            n_frames = cond['y']['lengths'].max()
            cond['y']['audio'] = cond['y']['audio'][:, :n_frames, :]
            # print(cond['y']['audio'].shape)
            # Unfolding - orig
            torch.manual_seed(123456789)
            sample = sample_fn(     # [1, 659, 1, 180]
                model,
                (1, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            motion = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()
            motion = np.multiply(motion, std) + mean
            rec_ric_data = recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), joints_num)[
                0]  # [1, 180, 55, 3]
            rec_ric_data = rec_ric_data.cpu().numpy()[0][:length]
            # print(rec_ric_data.shape)
            video_path = os.path.join(generated_save_path, "{}.mp4".format(name))

            plot_3d_motion_1(video_path,
                             kinematic_chain, np.array(rec_ric_data), captions,
                             fps=20, dataset='humanml', vis_mode='unfold_arb_len', handshake_size=0,
                             blend_size=0, step_sizes=step_sizes, lengths=cond['y']['lengths'])
            if Audio2Gesture:
                video_withaudio_path = os.path.join(generated_save_path, "{}_with_audio.mp4".format(name))
                add_audio_to_video_pydub(video_file=video_path, audio_tags=[[name.split('_')[0], name, 0, 180]],
                                         source_BEAT_path=source_BEAT_path,
                                         output_video_file=video_withaudio_path, merge_segments=[[0, 180]])
            np.save(os.path.join(generated_save_path, name + ".npy"), rec_ric_data)



if __name__ == "__main__":
    '''
    python -m sample.val_mdm --save_dir " "
    --batch_size 256 --n_frames 180 --n_seed 0
    --torch_manual_seed 123456789 --guidacnce_param 0
    '''

    version = 'v3_4'        # v2_0  ref
    model_name = "model000600000"


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join("/apdcephfs/private_yyyyyyyang/code/mdm/save", "my_" + version)
    save_path = os.path.join(save_path, model_name)
    print(save_path)

    model_path = os.path.join("/apdcephfs/private_yyyyyyyang/code/mdm/save/my_" + version + "/", model_name + ".pt")
    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, model_version=version, device=device, sample_mode=True)
    sample_fn = diffusion.p_sample_loop
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()

    mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/" + version.split('_')[0] + "_mean.npy")
    std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/"+ version.split('_')[0] +"_std.npy")

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()
