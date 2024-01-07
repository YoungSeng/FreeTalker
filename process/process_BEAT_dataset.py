import pdb
import argparse
import os
import glob
import numpy as np
import torch
import librosa
import torch.nn.functional as F
import sys
[sys.path.append(i) for i in ['./process']]
from tool import *


def wavlm_init(wavlm_model_path, device=torch.device('cuda:0')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))  # load the pre-trained checkpoints
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg


def wav2wavlm(model, wav_input_16khz, cfg, device=torch.device('cuda:0')):
    with torch.no_grad():
        wav_input_16khz = wav_input_16khz.to(device)
        if cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        wav_len = wav_input_16khz.shape[0]
        chunk_len = 16000 * 6
        num_chunks = wav_len // chunk_len + 1
        wav_input_16khz = torch.nn.functional.pad(wav_input_16khz, (0, chunk_len * num_chunks - wav_len))
        wav_input_16khz = wav_input_16khz.reshape(num_chunks, chunk_len)
        rep = []
        for i in range(0, num_chunks, 10):
            rep.append(model.extract_features(wav_input_16khz[i:i + 10])[0])
        rep = torch.cat(rep, dim=0)
        del wav_input_16khz
        rep = rep.reshape(-1, rep.shape[-1]).detach().cpu()
        return rep


def load_audio(audiofile, wavlm_model, cfg, device=torch.device('cuda:0')):
    wav, sr = librosa.load(audiofile, sr=16000)
    wav_input_16khz = torch.from_numpy(wav).to(torch.float32)
    # wav_input_16khz = torch.randn(1, 10000)     # (1, 10000) -> (1, 512, 1999) -> (1, 512, 999) -> (1, 512, 499) -> (1, 512, 249) -> (1, 512, 124), -> (1, 512, 62) -> (1, 512, 31)
    mfcc_f = calculate_mfcc(wav, sr)  # (7205, 40)
    melspec_f = calculate_spectrogram(wav, sr)  # (7205, 64)
    prosody = extract_prosodic_features(audiofile)  # (7199, 4)
    crop_length = min(mfcc_f.shape[0], melspec_f.shape[0], prosody.shape[0])
    wavlm_f = wav2wavlm(wavlm_model, wav_input_16khz, cfg, device)  # [12201, 1024]
    wavlm_f = F.interpolate(wavlm_f.unsqueeze(0).transpose(1, 2), size=crop_length, align_corners=True,
                            mode='linear').transpose(1, 2).squeeze(0)
    onsets_f, _ = extract_onsets(audiofile)
    # x = np.linspace(0, len(wav) - 1, num=len(wav))
    xp = np.linspace(0, len(wav) - 1, num=crop_length + 1)
    # audio_hfc = np.interp(xp, x, y)     # np.count_nonzero(audio_hfc)
    silence = np.array([0.] * len(wav))
    silence[(np.clip(onsets_f * 16000, 0, len(wav) - 1)).astype('int64')] = 1
    onsets_resample = np.array([0.] * crop_length)
    for i in range(1, crop_length + 1):
        onsets_resample[i - 1] = (max(silence[int(xp[i - 1]):int(xp[i])])) == 1
    audio_f = np.concatenate(
        (mfcc_f[:crop_length], melspec_f[:crop_length], prosody[:crop_length], wavlm_f, onsets_resample.reshape(-1, 1)),
        axis=1)
    return audio_f



def process_BEAT_wav(BEAT_source_PATH, BEAT_save_PATH, wavlm_model, cfg, device):
    # for speaker in ['2', '10']:     # 20230813

    for speaker in ['2', '4', '6', '8']:  # 20230901
        print(f"Processing {speaker}...")
        speaker_path = os.path.join(BEAT_source_PATH, speaker)
        speaker_save_path = os.path.join(BEAT_save_PATH, speaker)
        if not os.path.exists(speaker_save_path):
            os.makedirs(speaker_save_path)
        wav_files = sorted(glob.glob(speaker_path + "/*.wav"))
        for v_i, wav_file in enumerate(wav_files):
            name = os.path.split(wav_file)[1][:-4]
            print(f"Processing {v_i + 1}/{len(wav_files)}: {name}")
            # process audio
            if os.path.exists(os.path.join(speaker_save_path, name + ".npy")):
                print(f'audio {name} exist')
            else:
                wav = load_audio(wav_file, wavlm_model, cfg, device)
                np.save(os.path.join(speaker_save_path, name + ".npy"), wav)        # (len, 1133)


def process_BEAT_dataset(BEAT_smplx_path, save_path, version="v0"):
    print("Processing BEAT dataset...")
    total_frames = 0
    all_txt = []
    for speaker in os.listdir(BEAT_smplx_path):
        print(f"Processing {speaker}...")
        smplx_path = os.path.join(BEAT_smplx_path, speaker)
        smplx_files = sorted(glob.glob(smplx_path + "/*.npz"))
        for v_i, smplx_file in enumerate(smplx_files):
            name = os.path.split(smplx_file)[1][:-4]
            all_txt.append(name)
            print(f"Processing {v_i + 1}/{len(smplx_files)}: {name}")       # 1/33: 10_kieks_0_103_103, Processing 1/120: 2_scott_0_100_100
            # print(f"Total frames: {total_frames}")      # speaker 10 total frames: 110952 (30fps, 3698s, 1.03h), speaker 2 total frames: 470650 (30fps, 4.36h) 2_scott_1_6_6.npz打不开
    train_txt, val_txt, test_txt = train_val_test_split(all_txt, 0.8, 0.1, 0.1)
    write_HUMANML3D_txt(train_txt, val_txt, test_txt, save_path, version)       # Processing 130/130: 6_carla_1_9_9



def write_HUMANML3D_txt(train_txt, val_txt, test_txt, save_path, version="v0"):
    with open(os.path.join(save_path, version + "_train.txt"), "w") as f:
        for line in train_txt:
            f.write(line + "\n")
    with open(os.path.join(save_path, version + "_val.txt"), "w") as f:
        for line in val_txt:
            f.write(line + "\n")
    with open(os.path.join(save_path, version + "_test.txt"), "w") as f:
        for line in test_txt:
            f.write(line + "\n")
    with open(os.path.join(save_path, version + "_all.txt"), "w") as f:
        for line in train_txt + val_txt + test_txt:
            f.write(line + "\n")
    with open(os.path.join(save_path, version + "_train_val.txt"), "w") as f:
        for line in train_txt + val_txt:
            f.write(line + "\n")


def train_val_test_split(all_txt, train_ratio, val_ratio, test_ratio):
    import random
    random.shuffle(all_txt)
    train_txt = all_txt[:int(len(all_txt) * train_ratio)]
    val_txt = all_txt[int(len(all_txt) * train_ratio):int(len(all_txt) * (train_ratio + val_ratio))]
    test_txt = all_txt[int(len(all_txt) * (train_ratio + val_ratio)):]
    return train_txt, val_txt, test_txt


def process_HUMANML3D_dataset(HUMANML3D_smplx_path, save_path, version="v0"):
    print("Processing HUMANML3D dataset...")
    total_frames = 0

    all_txt = []

    smplx_files = sorted(glob.glob(HUMANML3D_smplx_path + "/*.npz"))
    for v_i, smplx_file in enumerate(smplx_files):
        name = os.path.split(smplx_file)[1][:-4]
        all_txt.append(name)
        print(f"Processing {v_i + 1}/{len(smplx_files)}: {name}")       # 1/33: 10_kieks_0_103_103, Processing 1/120: 2_scott_0_100_100
        # print(f"Total frames: {total_frames}")      # speaker 10 total frames: 110952 (30fps, 3698s, 1.03h), speaker 2 total frames: 470650 (30fps, 4.36h) 2_scott_1_6_6.npz打不开, 100个HUMANML3D 13696

    train_txt, val_txt, test_txt = train_val_test_split(all_txt, 0.8, 0.01, 0.19)
    write_HUMANML3D_txt(train_txt, val_txt, test_txt, save_path, version=version)


def clip_init(clip_model_path, device, clip_version='ViT-B/32'):
    import clip
    print(clip_version, clip_model_path)
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False, download_root=str(clip_model_path))  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model.to(device)


def process_HUMANML3D_txt(HUMANML3D_txt_save_path, HUMANML3D_save_path, clip_model, device, dataset='humanml', fps=20, version='v0'):
    import clip
    import codecs as cs
    def encode_text(raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts

        max_text_len = 20 if dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return clip_model.encode_text(texts.to(device)).float()

    HUMANML3D_text = os.path.join(HUMANML3D_txt_save_path, 'text_data', 'processed')
    if not os.path.exists(HUMANML3D_save_path):
        os.makedirs(HUMANML3D_save_path)
    HUMANML3D_items = sorted(glob.glob(HUMANML3D_text + "/*.txt"))
    for v_i, HUMANML3D_item in enumerate(HUMANML3D_items):
        print(f"Processing {HUMANML3D_item}, {v_i}/{len(HUMANML3D_items)}...")
        text_data = []
        with cs.open(HUMANML3D_item) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict['caption'] = caption
                text_dict['caption_clip'] = encode_text(caption).cpu().detach().numpy()
                text_dict['tokens'] = tokens

                if f_tag == 0.0 and to_tag == 0.0:
                    text_dict['clip_motion'] = False
                else:
                    text_dict['clip_motion'] = True
                    text_dict['begin'] = int(f_tag * fps)
                    text_dict['end'] = int(to_tag * fps)

                text_data.append(text_dict)
        np.savez(os.path.join(HUMANML3D_save_path, os.path.split(HUMANML3D_item)[1][:-4] + ".npz"), text_data=text_data)


def process_f5_file(BEAT_smplx_path, BEAT_save_PATH, BEAT_txt_save_PATH, HUMANML3D_smplx_path, HUMANML3D_save_path, HUMANML3D_txt_save_path, h5_save_path, down_sample_save_path, version='v0', step=1):
    import h5py
    if not os.path.exists(h5_save_path):
        os.makedirs(h5_save_path)
    if not os.path.exists(down_sample_save_path):
        os.makedirs(down_sample_save_path)

    def process(mode):      # 15/478
        print(f"Processing {mode}...")
        mode_save_path = os.path.join(down_sample_save_path, mode)
        with h5py.File(os.path.join(h5_save_path, version + '_' + mode + ".h5"), "w") as h5:     # "BEAT_HUMANML3D_"
            total_index = 0
            BEAT_items = []
            with open(os.path.join(BEAT_txt_save_PATH, version + '_' + mode + ".txt"), "r") as f:       # version.replace('v2', 'v1') + '_' +
                lines = f.readlines()
                for line in lines:
                    BEAT_items.append(line.strip())
            for v_i, BEAT_item in enumerate(BEAT_items):        # 30 fps

                print(f"Processing {BEAT_item}, {v_i}/{len(BEAT_items)}...")
                if version == 'v0':
                    BEAT_smplx_file = np.load(os.path.join(mode_save_path, 'motion', BEAT_item + ".npz"))
                    BEAT_smplx_poses = BEAT_smplx_file["poses"].reshape(len(BEAT_smplx_file["poses"]), -1)  # (len_smplx, 165)
                    BEAT_smplx_trans = BEAT_smplx_file["trans"]
                    BEAT_motion_feat = np.concatenate((BEAT_smplx_poses, BEAT_smplx_trans), axis=1)  # (len_smplx, 168)

                elif version == 'v1':
                    BEAT_motion_feat = np.load(os.path.join(mode_save_path, 'motion_joints_vecs', BEAT_item + ".npy"))

                elif version == 'v2':
                    BEAT_motion_feat = np.load(os.path.join(mode_save_path, 'motion_joints_vecs_2', BEAT_item + ".npy"))

                elif version == 'v3':
                    BEAT_motion_feat = np.load(os.path.join(mode_save_path, 'motion_joints_vecs_3', BEAT_item + ".npy"))


                BEAT_audio_feat = np.load(os.path.join(mode_save_path, 'audio', BEAT_item + ".npy"))     # (len_audio, 512)

                g_data = h5.create_group(str(total_index))
                if version == 'v0':
                    g_data.create_dataset("audio", data=BEAT_audio_feat, dtype=np.float32)
                elif version == 'v1' or version == 'v2' or version == 'v3':
                    # print(BEAT_audio_feat.shape, BEAT_motion_feat.shape)
                    if BEAT_audio_feat.shape[0] - 1 == BEAT_motion_feat.shape[0]:
                        g_data.create_dataset("audio", data=BEAT_audio_feat[1:], dtype=np.float32)
                    elif BEAT_audio_feat.shape[0] == BEAT_motion_feat.shape[0]:
                        g_data.create_dataset("audio", data=BEAT_audio_feat, dtype=np.float32)
                    else:
                        raise ValueError("Audio and motion length not match!")

                g_data.create_dataset("motion", data=BEAT_motion_feat, dtype=np.float32)
                # g_data.create_dataset("text", data=['unconstrained'])
                g_data.create_dataset("text", data=np.zeros(512), dtype=np.float32)
                g_data.create_dataset("dataset", data=['BEAT'])
                if mode == 'val':
                    g_data.create_dataset("caption", data=['unconstrained'])
                    g_data.create_dataset("name", data=[BEAT_item])
                total_index += 1


            HUMANML3D_items = []
            # if version == 'v3':
            #     with open(os.path.join(HUMANML3D_txt_save_path, 'v2' + '_' + mode + ".txt"), "r") as f:
            #         lines = f.readlines()
            #         for line in lines:
            #             HUMANML3D_items.append(line.strip())
            # else:
            #     with open(os.path.join(HUMANML3D_txt_save_path, version + '_' + mode + ".txt"), "r") as f:
            #         lines = f.readlines()
            #         for line in lines:
            #             HUMANML3D_items.append(line.strip())

            with open(os.path.join(HUMANML3D_txt_save_path, version + '_' + mode + ".txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    HUMANML3D_items.append(line.strip())

            error_files = []
            for v_i, HUMANML3D_item in enumerate(HUMANML3D_items):      # 20 fps
                print(f"Processing {HUMANML3D_item}, {v_i}/{len(HUMANML3D_items)}...")
                if version == 'v0':
                    HUMANML3D_smplx_file = os.path.join(HUMANML3D_smplx_path, HUMANML3D_item + ".npz")
                elif version == 'v1' or version == 'v2' or version == 'v3':
                    HUMANML3D_smplx_file = os.path.join(HUMANML3D_smplx_path, HUMANML3D_item + ".npy")
                try:
                    HUMANML3D_smplx = np.load(HUMANML3D_smplx_file)
                except:
                    print(HUMANML3D_smplx_path, f"Error: {HUMANML3D_item}")
                    error_files.append(HUMANML3D_item)
                    continue

                if version == 'v0':
                    HUMANML3D_smplx_poses = HUMANML3D_smplx["poses"].reshape(len(HUMANML3D_smplx["poses"]), -1)       # (len_smplx, 165)
                    HUMANML3D_smplx_trans = HUMANML3D_smplx["trans"] + np.concatenate([np.random.normal(loc=0, scale=0.15, size=(1, 2)), np.zeros((1, 1))], axis=1)     # random begin position
                    HUMANML3D_motion_feat = np.concatenate((HUMANML3D_smplx_poses, HUMANML3D_smplx_trans), axis=1)     # (len_smplx, 168)
                elif version == 'v1' or version == 'v2' or version == 'v3':
                    HUMANML3D_motion_feat = HUMANML3D_smplx

                HUMANML3D_text = np.load(os.path.join(HUMANML3D_save_path, HUMANML3D_item + ".npz"), allow_pickle=True)['text_data']
                for text_data in HUMANML3D_text:
                    if text_data['clip_motion']:
                        HUMANML3D_motion_feat_ = HUMANML3D_motion_feat[text_data['begin']:text_data['end']]
                    else:
                        HUMANML3D_motion_feat_ = HUMANML3D_motion_feat

                    len_smplx = HUMANML3D_motion_feat_.shape[0]
                    if len_smplx < 40 or len_smplx > 180:       # unfollow MDM
                        continue
                    g_data = h5.create_group(str(total_index))
                    g_data.create_dataset("motion", data=HUMANML3D_motion_feat_, dtype=np.float32)
                    # g_data.create_dataset("audio", data=['unconstrained'])

                    g_data.create_dataset("audio", data=np.zeros((len_smplx, 1133)), dtype=np.float32)
                    g_data.create_dataset("text", data=text_data['caption_clip'][0], dtype=np.float32)
                    g_data.create_dataset("length", data=len_smplx, dtype=np.float32)
                    g_data.create_dataset("dataset", data=['HUMANML3D'])
                    if mode == 'val':
                        g_data.create_dataset("caption", data=text_data['caption'][0])
                        g_data.create_dataset("name", data=[HUMANML3D_item])
                    total_index += 1
            print('error_files', error_files)      # ['M000787', '007870'], ['M000483', 'M000454', '007729']


    def downsample(mode):
        print(f"Dwonsampling {mode}...")
        mode_save_path = os.path.join(down_sample_save_path, mode)
        if not os.path.exists(mode_save_path):
            os.makedirs(mode_save_path)
        if not os.path.exists(os.path.join(mode_save_path, "audio")):
            os.makedirs(os.path.join(mode_save_path, "audio"))
        if not os.path.exists(os.path.join(mode_save_path, "motion")):
            os.makedirs(os.path.join(mode_save_path, "motion"))
        BEAT_items = []
        with open(os.path.join(BEAT_txt_save_PATH, version + '_' + mode + ".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                BEAT_items.append(line.strip())
        for v_i, BEAT_item in enumerate(BEAT_items):  # 30 fps
            speaker = BEAT_item.split("_")[0]
            print(f"Processing {BEAT_item}, {v_i}/{len(BEAT_items)}...")
            BEAT_smplx_file = os.path.join(BEAT_smplx_path, speaker, BEAT_item + ".npz")
            BEAT_smplx = np.load(BEAT_smplx_file)
            BEAT_smplx_poses = BEAT_smplx["poses"].reshape(len(BEAT_smplx["poses"]), -1)  # (len_smplx, 165)
            BEAT_smplx_trans = BEAT_smplx["trans"]  # (len_smplx, 3)
            BEAT_motion_feat = np.concatenate((BEAT_smplx_poses, BEAT_smplx_trans), axis=1)  # (len_smplx, 168)
            BEAT_audio_feat = np.load(
                os.path.join(BEAT_save_PATH, speaker, BEAT_item + ".npy"))  # (len_smplx, 1133)
            if version != 'v3':
                MIN_LEN = min(BEAT_motion_feat.shape[0], BEAT_audio_feat.shape[0])
                BEAT_motion_feat = BEAT_motion_feat[:MIN_LEN]
                BEAT_audio_feat = BEAT_audio_feat[:MIN_LEN]
                TARGET_LEN = MIN_LEN * 2 // 3
            else:
                TARGET_LEN = min(BEAT_motion_feat.shape[0], BEAT_audio_feat.shape[0] * 2 // 3)
                BEAT_motion_feat = BEAT_motion_feat[:TARGET_LEN]
                BEAT_audio_feat = BEAT_audio_feat[:TARGET_LEN * 3 // 2]

            # 30 fps to 20 fps
            if version == 'v0':
                BEAT_motion_feat = F.interpolate(torch.from_numpy(BEAT_motion_feat).unsqueeze(0).transpose(1, 2),
                                             size=TARGET_LEN, align_corners=True,
                                             mode='linear').transpose(1, 2).squeeze(0).detach().numpy()

            np.savez(os.path.join(mode_save_path, "motion", BEAT_item + ".npz"), poses=BEAT_motion_feat[:, :165],
                     trans=BEAT_motion_feat[:, 165:168],
                     gender="neutral", surface_model_type='smplx', mocap_frame_rate=20, betas=np.zeros(16))

            BEAT_audio_feat = F.interpolate(torch.from_numpy(BEAT_audio_feat).unsqueeze(0).transpose(1, 2),
                                            size=TARGET_LEN, align_corners=True,
                                            mode='linear').transpose(1, 2).squeeze(0).detach().numpy()
            np.save(os.path.join(mode_save_path, "audio", BEAT_item + ".npy"), BEAT_audio_feat)


    for mode in ['train', 'val']:     # 'train', 'val', 'test'
        if step == 1:
            if version == 'v0' or version == 'v3':
                downsample(mode)
        elif step == 2:
            process(mode)


def cal_statistics_mean(h5_save_path, version='v0'):
    import h5py
    h5 = h5py.File(os.path.join(h5_save_path, version + '_train' + ".h5"), 'r')
    motion_trn = [h5[key]['motion'][:] for key in h5.keys()]
    h5.close()
    print("Total trn clips:", len(motion_trn))

    h5 = h5py.File(os.path.join(h5_save_path, version + '_val' + ".h5"), 'r')
    motion_val = [h5[key]['motion'][:] for key in h5.keys()]
    h5.close()
    print("Total val clips:", len(motion_val))

    motion = np.vstack(motion_trn + motion_val)
    np.save(os.path.join(h5_save_path, version + '_mean.npy'), np.mean(motion, axis=0))
    # np.save(os.path.join(h5_save_path, version + '_std.npy'), np.std(motion, axis=0) + 1e-6)

def cal_statistics(h5_save_path, version='v0', batch_size=1000):
    import h5py
    def process_batch(motion_list):
        motion = np.vstack(motion_list)
        return np.var(motion, axis=0, ddof=1)

    def process_h5_file(file_path):
        motion_list = []
        var_list = []
        with h5py.File(file_path, 'r') as h5:
            for idx, key in enumerate(h5.keys()):
                print(f"Processing idx {idx}/{len(h5.keys())}...")
                motion_list.append(h5[key]['motion'][:])
                if (idx + 1) % batch_size == 0:
                    var_list.append(process_batch(motion_list))
                    motion_list = []
            if motion_list:
                var_list.append(process_batch(motion_list))
        return var_list

    train_file = os.path.join(h5_save_path, version + '_train' + ".h5")
    val_file = os.path.join(h5_save_path, version + '_val' + ".h5")

    var_train_list = process_h5_file(train_file)
    var_val_list = process_h5_file(val_file)

    var_list = var_train_list + var_val_list
    total_clips = sum([v.shape[0] for v in var_list])
    print("Total clips:", total_clips)

    weighted_var = np.zeros_like(var_list[0])
    for v in var_list:
        weight = v.shape[0] / total_clips
        weighted_var += weight * v

    weighted_std = np.sqrt(weighted_var + 1e-6)
    np.save(os.path.join(h5_save_path, version + '_std.npy'), weighted_std)


def process_AMASS_dataset(AMASS_smplx_path):
    '''
    # KIT_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/KIT/"
    # SSM_smplx_path = "/apdcephfs/private_yyyyyyyang/data/SSM/"
    # HumanEva_smplx_path = "/apdcephfs/private_yyyyyyyang/data/HumanEva/"
    # process_AMASS_dataset(KIT_smplx_path)
    '''
    print("Processing KIT dataset...")
    for number in os.listdir(AMASS_smplx_path):
        print(f"Processing {number}...")
        smplx_path = os.path.join(AMASS_smplx_path, number)
        smplx_files = sorted(glob.glob(smplx_path + "/*.npz"))
        for v_i, smplx_file in enumerate(smplx_files):
            name = os.path.split(smplx_file)[1][:-4]
            print(f"Processing {v_i + 1}/{len(smplx_files)}: {name}")

            # process motion
            smplx = np.load(smplx_file)
            pdb.set_trace()


def check_nan(fold):
    for item in os.listdir(fold):
        if item.endswith('.npy'):
            print('checking', item)
            try:
                npy_file = np.load(os.path.join(fold, item))
            except:
                print(item, "error")
                return True
            if np.isnan(npy_file).any():
                pdb.set_trace()
                print(item, "nan", npy_file)
                return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default="prepare")
    parser.add_argument('--BEAT_smplx_path', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_smplx")
    parser.add_argument('--BEAT_txt_save_PATH', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/")
    parser.add_argument('--wavlm_model_path', type=str, default="/apdcephfs/private_yyyyyyyang/pre-trained-models/WavLM-Large.pt")
    parser.add_argument('--BEAT_source_PATH', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/")
    parser.add_argument('--BEAT_save_PATH', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_wav_feat")
    parser.add_argument('--h5_save_path', type=str, default="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data")
    parser.add_argument('--down_sample_save_path', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/my_downsample/")
    parser.add_argument('--HUMANML3D_smplx_path', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs_2/")
    parser.add_argument('--HUMANML3D_save_path', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/" + 'v2' + "_HUMANML3D_txt_feat/")
    parser.add_argument('--HUMANML3D_txt_save_path', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/")
    parser.add_argument('--clip_model_path', type=str, default="/apdcephfs/private_yyyyyyyang/clip/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    pip install librosa essentia pydub praat-parselmouth
    python -m process.process_BEAT_dataset
    '''

    device = torch.device("cuda:0")

    '''
    BEAT_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/"
    BEAT_source_PATH = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"
    BEAT_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/BEAT_wav_feat"
    BEAT_txt_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/"
    wavlm_model_path = "/apdcephfs/private_yyyyyyyang/pre-trained-models/WavLM-Large.pt"
    # wavlm_model, cfg = wavlm_init(wavlm_model_path, device)
    # process_BEAT_wav(BEAT_source_PATH, BEAT_save_PATH, wavlm_model, cfg, device)
    # process_BEAT_dataset(BEAT_smplx_path, BEAT_txt_save_PATH)

    HUMANML3D_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/"
    HUMANML3D_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/HUMANML3D_txt_feat/"
    HUMANML3D_txt_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/"
    clip_model_path = "/apdcephfs/private_yyyyyyyang/clip/"
    # clip_model = clip_init(clip_model_path, device)
    # process_HUMANML3D_txt(HUMANML3D_txt_save_path, HUMANML3D_save_path, clip_model, device)
    # process_HUMANML3D_dataset(HUMANML3D_smplx_path, HUMANML3D_txt_save_path)        # 4781


    h5_save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data"
    down_sample_save_path = "/apdcephfs/share_1290939/new_data/BEAT/downsample/"
    # process_f5_file(BEAT_smplx_path, BEAT_save_PATH, BEAT_txt_save_PATH, HUMANML3D_smplx_path, HUMANML3D_save_path, HUMANML3D_txt_save_path, h5_save_path, down_sample_save_path)
    cal_statistics(h5_save_path)
    '''

    '''
    BEAT_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/"
    BEAT_source_PATH = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"
    BEAT_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/BEAT_wav_feat"
    BEAT_txt_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/"
    wavlm_model_path = "/apdcephfs/private_yyyyyyyang/pre-trained-models/WavLM-Large.pt"
    # wavlm_model, cfg = wavlm_init(wavlm_model_path, device)
    # process_BEAT_wav(BEAT_source_PATH, BEAT_save_PATH, wavlm_model, cfg, device)
    # process_BEAT_dataset(BEAT_smplx_path, BEAT_txt_save_PATH)

    HUMANML3D_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs/"
    HUMANML3D_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/HUMANML3D_txt_feat/"
    HUMANML3D_txt_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/"
    clip_model_path = "/apdcephfs/private_yyyyyyyang/clip/"
    # clip_model = clip_init(clip_model_path, device)
    # process_HUMANML3D_txt(HUMANML3D_txt_save_path, HUMANML3D_save_path, clip_model, device)
    process_HUMANML3D_dataset(HUMANML3D_smplx_path, HUMANML3D_txt_save_path)        # 4781


    h5_save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data"
    down_sample_save_path = "/apdcephfs/share_1290939/new_data/BEAT/downsample/"

    # process_f5_file(BEAT_smplx_path, BEAT_save_PATH, BEAT_txt_save_PATH, HUMANML3D_smplx_path, HUMANML3D_save_path, HUMANML3D_txt_save_path, h5_save_path, down_sample_save_path, version=version)
    # cal_statistics(h5_save_path, version=version)

    # check_nan("/apdcephfs/share_1290939/new_data/BEAT/downsample/train/motion_joints_vecs")
    '''

    '''
    version = "v2"
    HUMANML3D_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/"
    HUMANML3D_txt_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/"
    # process_HUMANML3D_dataset(HUMANML3D_smplx_path, HUMANML3D_txt_save_path, version)  # 4781

    HUMANML3D_save_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/" + version + "_HUMANML3D_txt_feat/"
    clip_model_path = "/apdcephfs/private_yyyyyyyang/clip/"
    # clip_model = clip_init(clip_model_path, device)
    # process_HUMANML3D_txt(HUMANML3D_txt_save_path, HUMANML3D_save_path, clip_model, device, version)

    h5_save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data"
    down_sample_save_path = "/apdcephfs/share_1290939/new_data/BEAT/downsample/"

    HUMANML3D_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs_2/"
    BEAT_smplx_path = "/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/"
    BEAT_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/BEAT_wav_feat"
    BEAT_txt_save_PATH = "/apdcephfs/share_1290939/new_data/BEAT/"

    process_f5_file(BEAT_smplx_path, BEAT_save_PATH, BEAT_txt_save_PATH, HUMANML3D_smplx_path, HUMANML3D_save_path,
                    HUMANML3D_txt_save_path, h5_save_path, down_sample_save_path, version=version)
    cal_statistics(h5_save_path, version=version)

    # Total trn clips: 33483
    # Total val clips: 457

    # check_nan("/apdcephfs/share_1290939/new_data/BEAT/downsample/train/motion_joints_vecs")
    '''

    args = get_args()
    version = "v3"
    if args.step == "prepare":
        BEAT_smplx_path = args.BEAT_smplx_path
        BEAT_txt_save_PATH = args.BEAT_txt_save_PATH
        process_BEAT_dataset(BEAT_smplx_path, BEAT_txt_save_PATH, version=version)

        HUMANML3D_smplx_path = args.HUMANML3D_smplx_path
        HUMANML3D_txt_save_path = args.HUMANML3D_txt_save_path
        process_HUMANML3D_dataset(HUMANML3D_smplx_path, HUMANML3D_txt_save_path, version)  # 4781

        wavlm_model_path = args.wavlm_model_path
        BEAT_source_PATH = args.BEAT_source_PATH
        BEAT_save_PATH = args.BEAT_save_PATH
        wavlm_model, cfg = wavlm_init(wavlm_model_path, device)
        process_BEAT_wav(BEAT_source_PATH, BEAT_save_PATH, wavlm_model, cfg, device)

        HUMANML3D_save_path = args.HUMANML3D_save_path
        clip_model_path = args.clip_model_path
        clip_model = clip_init(clip_model_path, device)
        process_HUMANML3D_txt(HUMANML3D_txt_save_path, HUMANML3D_save_path, clip_model, device, version)

        h5_save_path = args.h5_save_path
        down_sample_save_path = args.down_sample_save_path

        # downsample audio and remove motion
        process_f5_file(BEAT_smplx_path, BEAT_save_PATH, BEAT_txt_save_PATH,
                        None, None, None,
                        h5_save_path, down_sample_save_path, version=version, step=1)
        # smplx to position     python process/process_SMPLX.py

    elif args.step == "generate_h5_file":

        # position to motion feature    python process/motion_representation.py

        HUMANML3D_smplx_path = args.HUMANML3D_smplx_path
        HUMANML3D_save_path = args.HUMANML3D_save_path
        HUMANML3D_txt_save_path = args.HUMANML3D_txt_save_path

        process_f5_file(args.BEAT_smplx_path, args.BEAT_save_PATH, args.BEAT_txt_save_PATH,
                        HUMANML3D_smplx_path, HUMANML3D_save_path, HUMANML3D_txt_save_path,
                        args.h5_save_path, args.down_sample_save_path, version=version, step=2)

        print("generate_h5_file is done, start to generate statistics")
        cal_statistics_mean(args.h5_save_path, version=version)
        cal_statistics(args.h5_save_path, version=version)

    print(args.step, "is done")
