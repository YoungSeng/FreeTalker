import pdb
import h5py
import numpy as np
import os
import subprocess
import copy
import torch
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len
import sys
import argparse

[sys.path.append(i) for i in ['./process']]
from process.plot_script import plot_3d_motion as plot_3d_motion_1
from process.motion_representation import plot_3d_motion as plot_3d_motion_2
from process.motion_representation import kinematic_chain, recover_from_ric, joints_num
from process.merge_mp4_audio import add_audio_to_video_pydub
import clip


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
    audio_tag = []

    for audio_file in sample_audio_files:
        speaker = audio_file.split("_")[0]
        audio_feat = np.load(os.path.join(BEAT_wav_feat, speaker, audio_file + ".npy"))
        audio_feat = audio_feat[:segment_length]
        sample_audio_feat.append(audio_feat)
        audio_tag.append([speaker, audio_file, 0, segment_length])
    for text_file in sample_text_files:
        text_feat = np.load(os.path.join(HUMANML3D_text_feat, text_file + ".npz"), allow_pickle=True)['text_data'][0]
        text.append(text_feat['caption'])
        text_feat = text_feat['caption_clip']
        sample_text_feat.append(text_feat)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        subprocess.call(["cp", "-r", os.path.join(GT_HUMANML3D_motion, text_file + ".npz"), save_path])
    print(text)
    return sample_audio_feat, sample_text_feat, text, audio_tag


def post_process(position, length_tag, handshake_size=20):
    """
    :param rec_ric_data: [length, 55, 3] e.g. [360, 55, 3]
    :param length_tag: where to cut, e.g. [120, 160, 120]
    :param handshake_size: 20
    :return: handle root shift and blend
    """
    blend_size = handshake_size

    import numpy as np
    position = np.array(position)
    # Initialize the result array
    result = np.zeros_like(position)

    # Calculate the starting indices of each segment
    # start_indices = [0] + [sum(length_tag[:i]) - blend_size * (2 * i - 1) for i in range(1, len(length_tag))] + [position.shape[0]]
    start_indices = [0] + [sum(length_tag[:i]) - blend_size * i for i in range(1, len(length_tag))] + [
        position.shape[0]]
    print('length tag: ', length_tag, 'start indices:', start_indices)

    # Process each segment
    for i, length in enumerate(length_tag):
        start = start_indices[i]
        end = start_indices[i + 1]

        if i == 0:
            # Copy the current segment to the result array
            result[start:end] = position[start:end]

        # Handle root shift
        elif i > 0:
            # print('start: ', start, 'end: ', end)

            # prev_blend = position[start:start+blend_size].copy()  # [10, 55, 3]
            #
            # print('prev blend last root: ', prev_blend[0][0])
            # # Calculate the root shift between the current and previous segments
            # root_shift = position[start + blend_size][0] - prev_blend[0][0]
            # print('current root: ', position[start + blend_size, 0])
            # print('root shift: ', root_shift)
            #
            # # Apply the root shift to the current segment
            # this_pose = position[start:end].copy() - root_shift
            #
            # print('this blend root: ', this_pose[10][0])  # they should be the close
            # # Blend the end of the previous segment and the start of the current segment
            # this_blend = this_pose[:blend_size]
            #
            # for j in range(blend_size):
            #     prev = prev_blend[j]
            #     next = this_blend[j]
            #     result[start + j] = prev * (blend_size - j) / (blend_size + 1) + next * (j + 1) / (blend_size + 1)

            # result[start + blend_size:end] = this_pose[blend_size:]

            prev_last = position[start][np.newaxis, :, :]

            pre_root = position[start][0]

            root_shift = position[start + blend_size][0] - pre_root

            position[start + blend_size:] = (position[start + blend_size:] - root_shift.reshape(1, 1, 3))

            next_first = position[start + blend_size][np.newaxis, :, :]
            result[start:start + blend_size] = prev_last * (blend_size - np.arange(blend_size))[:, np.newaxis,
                                                           np.newaxis] / (blend_size + 1) + next_first * (np.arange(
                blend_size) + 1)[:, np.newaxis, np.newaxis] / (blend_size + 1)
            result[start + blend_size:end] = position[start + blend_size:end]


    return result



# Test the function with sample data
# rec_ric_data = np.random.rand(360, 55, 3)
# length_tag = [120, 160, 120]
# result = post_process(rec_ric_data, length_tag)
#
# print(result.shape)  # Should output (360, 55, 3)


def clip_init(clip_model_path, device, clip_version='ViT-B/32'):
    print(clip_version, clip_model_path)
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False, download_root=str(clip_model_path))  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model.to(device)


def encode_text(clip_model, raw_text):
    max_text_len = 20
    default_context_length = 77
    context_length = max_text_len + 2 # start_token + 20 + end_token
    assert context_length < default_context_length
    texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
    zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
    texts = torch.cat([texts, zero_pad], dim=1)
    return clip_model.encode_text(texts.to(device)).float()


def vis_controls(guidacnce_param=0):
    clip_model_path = args_.clip_model_path
    device = torch.device("cuda:0")
    clip_model = clip_init(clip_model_path, device)
    # feat = encode_text(clip_model, 'a person walks forward.')

    def sequence_1():

        test_audio_feat_1 = np.load("./data/BEAT/my_downsample/val/audio/2_scott_0_14_14.npy")  # (619, 1133) 20fps
        test_audio_feat_2 = np.load("./data/BEAT/my_downsample/val/audio/2_scott_0_37_37.npy")
        test_feat_list = [test_audio_feat_1[:segment_length - 20],
                          test_audio_feat_1[segment_length - 20:segment_length * 2 - 40],
                          test_audio_feat_2[:segment_length - 20],
                          test_audio_feat_2[segment_length - 20:segment_length * 2 - 40]]
        test_feat_list = [np.expand_dims(np.vstack([np.zeros_like(i)[:10], i, np.zeros_like(i)[:10]]), axis=0) for i in
                          test_feat_list]

        text_list = ['a person looks at their wrist.', 'a person jogs in place.', 'a person waves with both arms above their head.']
        text_length = [100, 90, 100]

        audio_tags = [['2', '2_scott_0_14_14', 0, 110 * 2],
                      ['2', '2_scott_0_37_37', 0, 110 * 2]]

        merge_segments = [[90, 90 + 110 * 2], [380, 380 + 110 * 2]]

        return test_feat_list, text_list, text_length

    def sequence_2():
        test_audio_feat_1 = np.load("./data/BEAT/my_downsample/val/audio/4_lawrence_0_19_19.npy")  # (619, 1133) 20fps
        test_audio_feat_2 = np.load("./data/BEAT/my_downsample/val/audio/4_lawrence_0_35_35.npy")
        test_feat_list = [test_audio_feat_1[:segment_length - 20],
                          test_audio_feat_1[segment_length - 20:segment_length * 2 - 40],
                          test_audio_feat_2[:segment_length - 20],
                          test_audio_feat_2[segment_length - 20:segment_length * 2 - 40]]
        test_feat_list = [np.expand_dims(np.vstack([np.zeros_like(i)[:10], i, np.zeros_like(i)[:10]]), axis=0) for i in
                          test_feat_list]

        text_list = ['a person crossed their arms.', 'a person jumps and moves their arms and legs outward and then inward.',
                     'the person is jumping rope.']
        text_length = [100, 90, 100]

        audio_tags = [['4', '4_lawrence_0_19_19', 0, 110 * 2],
                      ['4', '4_lawrence_0_35_35', 0, 110 * 2]]

        merge_segments = [[90, 90 + 110 * 2], [380, 380 + 110 * 2]]

        return test_feat_list, text_list, text_length

    test_feat_list, text_list, text_length = sequence_1()
    # test_feat_list, text_list, text_length = sequence_2()

    text_feat = [encode_text(clip_model, i).cpu().detach().numpy() for i in text_list]

    control_text = text_list[:1] + ["Audio2Gesture"] * 2 + text_list[1:2] + ["Audio2Gesture"] * 2 +  text_list[2:3]

    control_text_feat = text_feat[:1] + [np.zeros((1, 512))] * 2 + text_feat[1:2] + [np.zeros((1, 512))] * 2 + text_feat[2:3]
    control_audio = [np.zeros((1, segment_length, 1133))] + test_feat_list[0:2] + [np.zeros((1, segment_length, 1133))] \
                    + test_feat_list[2:4] + [np.zeros((1, segment_length, 1133))]
    control_length = text_length[:1] + [segment_length] * 2 + text_length[1:2] + [segment_length] * 2 + text_length[2:3]

    control_audio_ = [np.zeros((1, segment_length, 1133))] * 7

    print("guidacnce_param: ", guidacnce_param)  # 0.35
    guidances = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * guidacnce_param

    control_dict = {'y': {
        'mask': torch.ones((len(control_length), 1, 1, segment_length)),  # 196 is humanml max frames number
        'lengths': torch.tensor(control_length),
        'text': torch.tensor(control_text_feat).squeeze(1).float(),
        'text_tag': text_list,
        'audio': torch.tensor(control_audio).squeeze(1).float(),
        'tokens': [''],
        'scale': guidances,
        'audio_': torch.tensor(control_audio_).squeeze(1).float(),
    }}

    captions = (
        [control_text[0]] * (control_length[0] - 10),
        *([text] * (length - 20) for text, length in zip(control_text[1:-1], control_length[1:-1])),
        [control_text[-1]] * (control_length[-1] - 10)
    )

    captions = sum(captions, [])  # 将嵌套列表展平

    return control_dict, captions, guidacnce_param


def customized_controls(guidacnce_param=0):
    clip_model_path = args_.clip_model_path
    device = torch.device("cuda:0")
    clip_model = clip_init(clip_model_path, device)
    # feat = encode_text(clip_model, 'a person walks forward.')

    test_audio_feat = np.load("./data/BEAT/my_downsample/val/audio/2_scott_0_55_55.npy")     # (619, 1133) 20fps
    test_feat_list = [test_audio_feat[:segment_length-20], test_audio_feat[segment_length-20:segment_length*2-40], test_audio_feat[segment_length*2-40:segment_length*3-60], test_audio_feat[segment_length*3:segment_length*4]]
    test_feat_list = [np.expand_dims(np.vstack([np.zeros_like(i)[:10], i, np.zeros_like(i)[:10]]), axis=0) for i in test_feat_list]

    def list_multiply(param, list1):
        return [param * i for i in list1]

    # text_list = ['a person accelerates forward then slows down.', 'a person raises right hand.',
    #              'a person bows.', 'a person turns around, walks.']
    # text_length = [120, 120, 120, 120]
    # text_feat = [encode_text(clip_model, i).cpu().detach().numpy() for i in text_list]
    #
    # control_text = text_list[:2] + ['Audio2Gesture'] * 2 + text_list[2:]
    # control_text_feat = text_feat[:2] + [np.zeros((1, 512))] * 2 + text_feat[2:]
    # control_audio = [np.zeros((1, segment_length, 1133))] * 2 + test_feat_list[:2] + [np.zeros((1, segment_length, 1133))] * 2        # segment_length
    # control_length = text_length[:2] + [segment_length] * 2 + text_length[2:]

    gamma_text = 1
    gamma_audio = 1       # 0.45

    text_list = ['a person runs forward then slows down.', 'a person raises right hand.',
                 'a person with his left hand raised.', 'a person raises right arm.',
                 'a person bows.', 'a person turns around']     #  then walks.
    # a person turns around and walks. a person walks toward the back 'a man turns to the back left and walks.'
    text_length = [120, 90, segment_length, segment_length, 80, 70]

    text_feat = [encode_text(clip_model, i).cpu().detach().numpy() for i in text_list]

    # control_text = text_list[:2] + [r"Audio2Gesture" + "\n" + r'a person with his left hand raised high.'] + ['Audio2Gesture'] + \
    #                [r'Audio2Gesture' + '\n' + r'a person raises right hand.'] + text_list[-2:]
    control_text = text_list[:2] + ["Audio2Gesture"] + ['Audio2Gesture'] + \
                   ['Audio2Gesture'] + text_list[-2:]

    control_text_feat = text_feat[:2] + list_multiply(gamma_text, text_feat[2:3]) + [np.zeros((1, 512))] + list_multiply(gamma_text, text_feat[3:4]) + text_feat[4:]
    control_audio = [np.zeros((1, segment_length, 1133))] * 2 + list_multiply(gamma_audio, test_feat_list[0:1]) + test_feat_list[1:2] + \
                    list_multiply(gamma_audio, test_feat_list[2:3]) + [np.zeros((1, segment_length, 1133))] * 2        # segment_length
    control_audio_ = [np.zeros((1, segment_length, 1133))] * 3 + test_feat_list[1:2] + \
                     [np.zeros((1, segment_length, 1133))] * 3
    control_length = text_length[:3] + [segment_length] + text_length[3:]
    # 110, 70, 105, 105, 105, 80, 110

    # guidances = torch.ones(len(control_length)) * 2.5#

    print("guidacnce_param: ", guidacnce_param)     # 0.35
    # guidances = torch.tensor([1.0, 1.0, 2.5, 1.0, 2.5, 1.0, 1.0])  # 1.0, 1.0, 2.5, 1.0, 2.5, 1.0, 1.0
    guidances = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * guidacnce_param


    control_dict = {'y': {
        'mask': torch.ones((len(control_length), 1, 1, segment_length)),  # 196 is humanml max frames number
        'lengths': torch.tensor(control_length),
        'text': torch.tensor(control_text_feat).squeeze(1).float(),
        'text_tag': text_list,
        'audio': torch.tensor(control_audio).squeeze(1).float(),
        'tokens': [''],
        'scale': guidances,
        'audio_': torch.tensor(control_audio_).squeeze(1).float(),
    }}

    captions = (
        [control_text[0]] * (control_length[0] - 10),
        *([text] * (length - 20) for text, length in zip(control_text[1:-1], control_length[1:-1])),
        [control_text[-1]] * (control_length[-1] - 10)
    )

    captions = sum(captions, [])  # 将嵌套列表展平

    # captions = [text_list[0]] * (text_length[0] - 10) + [text_list[1]] * (text_length[1] - 20) + [text_list[2]] * (text_length[2] - 20) + [text_list[-1]] * (text_length[-1] - 10)

    return control_dict, captions, guidacnce_param


def process(sample_audio_feat, sample_text_feat, segment_length, save_path, model_name, n_seed=10, version="v0", text=''):
    from utils.model_util import create_model_and_diffusion
    from utils.model_util import load_model_wo_clip
    from utils.parser_util import train_args
    args = train_args()

    # text2motion

    # text1 = sample_text_feat[0]
    # aux = np.load("/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/10/10_kieks_0_12_12.npz", allow_pickle=True)      # note: 30fps
    # if 'v0' in version:
    #     aux = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/005204.npz", allow_pickle=True)
    #     seed1 = np.concatenate((aux['poses'].reshape(aux['poses'].shape[0], -1), aux['trans']), axis=1)[:n_seed]
    # elif 'v1' in version:
    #     aux = np.load("/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx_joints_vecs/005204.npy")
    #     seed1 = aux[:n_seed]

    mean = np.load("./data/prcocessed_data/" + version.split('_')[0] + "_mean.npy")
    std = np.load("./data/prcocessed_data/"+ version.split('_')[0] +"_std.npy")
    # if 'v0' in version or 'v1' in version:
    #     seed1 = (seed1 - mean) / std
    #     seed1 = torch.from_numpy(seed1).to(device).float().unsqueeze(0).unsqueeze(0).permute(1, 3, 0, 2)

    model_path = os.path.join("./save/my_" + version + "/", model_name + ".pt")
    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, model_version=version, device=device, sample_mode=True)
    sample_fn = diffusion.p_sample_loop
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()
    model = ClassifierFreeSampleModel(model)

    # text2 = sample_text_feat[1]
    # audio2 = np.expand_dims(sample_audio_feat[0], axis=0)

    if 'ref' in version:
        model_kwargs = {'y': {
            'mask': torch.ones((2, 1, 1, segment_length)),      # 196 is humanml max frames number
            'lengths': torch.tensor([120, 120]),
            'text': ['a person walks, speeds up, and jumps.', 'a person walks forward while swinging their arms'],
            'audio': torch.tensor([np.zeros((1, 120, 1133)), np.zeros((1, 120, 1133))]).squeeze(1).float(),  # [:, :170],
            'tokens': [''],
            'scale': torch.ones(len([120, 120])) * 2.5
        }}
    else:
        if args.vis_mode == 'customized_controls':
            model_kwargs, captions, guidacnce_param = customized_controls(args.guidacnce_param)
        elif args.vis_mode == 'vis_controls':
            model_kwargs, captions, guidacnce_param = vis_controls(args.guidacnce_param)

    handshake_size = 20
    blend_len = 10

    model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in
                         model_kwargs['y'].items()}

    max_arb_len = model_kwargs['y']['lengths'].max()  # [100, 120,  95, 123]

    min_arb_len = 2 * handshake_size + 2 * blend_len + 10  # 20, 10 -> 70

    torch.manual_seed(args.torch_manual_seed)
    samples_per_rep_list, samples_type, orig_sample = double_take_arb_len(args, diffusion, model, model_kwargs, max_arb_len, guidacnce_param)     # [3, 659, 1, 160],

    '''
    print(orig_sample[0].shape)     # [3, 659, 1, 125]
    orig_sample = orig_sample[0].squeeze(2).permute(0, 2, 1).cpu().numpy()
    print(orig_sample.shape)        # [3, 125, 659]
    for iii in range(len(orig_sample)):
        joints_num = 55
        np.save(os.path.join(save_path, "positions_real_{}.npy".format(iii)), orig_sample[iii])
        rec_ric_data = recover_from_ric(torch.from_numpy(np.multiply(orig_sample[iii], std) + mean).unsqueeze(0).float(), joints_num)[0]
        print("rec_ric_data shape: ", rec_ric_data.shape)
        # rec_ric_data = post_process(rec_ric_data, length_tag=model_kwargs['y']['lengths'].cpu().numpy(), handshake_size=20)
        plot_3d_motion_1(os.path.join(save_path, "positions_real_{}.mp4".format(iii)), kinematic_chain, np.array(rec_ric_data),
                         captions,
                         fps=20, dataset='humanml', vis_mode='gt')
    '''


    step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
    for ii, len_i in enumerate(model_kwargs['y']['lengths']):
        if ii == 0:
            step_sizes[ii] = len_i
            continue
        step_sizes[ii] = step_sizes[ii - 1] + len_i - handshake_size

    final_n_frames = step_sizes[-1]

    for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

        out_list = unfold_sample_arb_len(sample_i, handshake_size, step_sizes, final_n_frames, model_kwargs)      # 120+160+120-40

    out_list = [i.detach().data.cpu().numpy() for i in out_list]

    sampled_seq = np.array(out_list).squeeze(2).transpose(0, 2, 1)      # (1, len, 168)
    sampled_seq = sampled_seq[:, n_seed:]

    out_poses = np.multiply(sampled_seq[0], std) + mean

    if "v0" in version:
        np.savez(os.path.join(save_path, "result.npz"), gender="neutral", surface_model_type='smplx', mocap_frame_rate=20,
                 poses=out_poses[:, :165], trans=out_poses[:, 165:168], betas=np.zeros(16))
    elif "v1" in version or "v2" in version or "v3" in version:

        # from process.motion_representation import plot_3d_motion, kinematic_chain, recover_from_ric, joints_num
        # rec_ric_data = recover_from_ric(torch.from_numpy(out_poses).unsqueeze(0).float(), joints_num)[0]
        # print("rec_ric_data shape: ", rec_ric_data.shape)
        # plot_3d_motion(os.path.join(save_path, "positions.mp4"), kinematic_chain, np.array(rec_ric_data), 'title', fps=20)

        joints_num = 55
        rec_ric_data = recover_from_ric(torch.from_numpy(out_poses).unsqueeze(0).float(), joints_num)[0]
        print("rec_ric_data shape: ", rec_ric_data.shape)
        # rec_ric_data = post_process(rec_ric_data, length_tag=model_kwargs['y']['lengths'].cpu().numpy(), handshake_size=20)
        plot_3d_motion_1(os.path.join(save_path, "positions_vis1_{}_{}.mp4".format(guidacnce_param, args.vis_mode)), kinematic_chain, np.array(rec_ric_data), captions,
                       fps=20, dataset='humanml', vis_mode='unfold_arb_len', handshake_size=20,
                       blend_size=10, step_sizes=step_sizes, lengths=model_kwargs['y']['lengths'])
        plot_3d_motion_2(os.path.join(save_path, "positions_vis2_{}_{}.mp4".format(guidacnce_param, args.vis_mode)), kinematic_chain, np.array(rec_ric_data), 'title', fps=20)

        # np.save(os.path.join(save_path, "result.npy"), out_poses)
        out_put_rec_ric_path = os.path.join(save_path, "result_rec_{}.npy".format(guidacnce_param))
        np.save(out_put_rec_ric_path, rec_ric_data)

        # subprocess.run(['python', '../human_body_prior/tutorials/mdm_motion2smpl.py', '--input', out_put_rec_ric_path, '--output', out_put_rec_ric_path[:-4] + '_smplx.npz'])

    elif 'ref' in version:
        import sys

        t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                               [9, 13, 16, 18, 20]]

        joints_num = 22
        rec_ric_data = recover_from_ric(torch.from_numpy(out_poses).unsqueeze(0).float(), joints_num)[0]
        print("rec_ric_data shape: ", rec_ric_data.shape)
        plot_3d_motion_1(os.path.join(save_path, "positions_real.mp4"), t2m_kinematic_chain, np.array(rec_ric_data), 'title',
                       fps=20, dataset='humanml', vis_mode='unfold_arb_len', handshake_size=20,
                       blend_size=10, step_sizes=step_sizes, lengths=model_kwargs['y']['lengths'])

        plot_3d_motion_2(os.path.join(save_path, "positions_my.mp4"), t2m_kinematic_chain, np.array(rec_ric_data), 'title', fps=20)

        np.save(os.path.join(save_path, "result.npy"), out_poses)

    print("Done!")

    # audio2motion


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--BEAT_text_path', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/")
    # parser.add_argument('--HUMANML3D_text_path', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/")
    parser.add_argument('--BEAT_wav_feat', type=str, default="/apdcephfs/share_1290939/new_data/BEAT/BEAT_wav_feat/")
    parser.add_argument('--HUMANML3D_text_feat', type=str, default="/apdcephfs/share_1290939/new_data/dataset/HumanML3D/HUMANML3D_txt_feat/")
    parser.add_argument('--model_name', type=str, default="model000600000")
    parser.add_argument("--vis_mode", default='customized_controls', type=str)
    parser.add_argument('--save_dir', type=str, default=" ")
    parser.add_argument("--guidacnce_param", default=0, type=float)
    parser.add_argument("--torch_manual_seed", default=1234567, type=int)
    parser.add_argument("--clip_model_path", default="/ceph/hdd/yangsc21/Python/mdm/data/clip", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
python -m sample.double_take --save_dir '' --guidacnce_param 1 --torch_manual_seed 123456 --model_name model000200000 --BEAT_wav_feat /ceph/datasets/BEAT/my_wav_feat/ --HUMANML3D_text_feat /ceph/datasets/SMPLX/HumanML3D/v3_HUMANML3D_txt_feat/ --vis_mode 
    '''
    # motion, audio, text, dataset_count = read_data()

    args_ = get_args()
    # BEAT_text_path = args.BEAT_text_path
    # HUMANML3D_text_path = args.HUMANML3D_text_path
    n_sample_audio = 1
    n_sample_text = 2

    # sample_audio_files, sample_text_files = pick_data(BEAT_text_path, HUMANML3D_text_path, n_sample_audio, n_sample_text)
    sample_audio_files, sample_text_files = None, None
    # print(sample_audio_files, sample_text_files)

    BEAT_wav_feat = args_.BEAT_wav_feat
    HUMANML3D_text_feat = args_.HUMANML3D_text_feat
    version = 'v3_0'        # v2_0  ref
    model_name = args_.model_name

    if 'v2' in version:
        segment_length = 120        # 160
        # segment_length = 125      # local attention Shape mismatch, 125 != 120
    elif 'v3' in version:
        segment_length = 130
    else:
        segment_length = 190

    save_path = os.path.join("./save", "my_" + version)
    save_path = os.path.join(save_path, model_name)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # GT_HUMANML3D_motion = "/apdcephfs/share_1290939/new_data/dataset/HumanML3D/vposer_smplx/"
    # "/apdcephfs/share_1290939/new_data/dataset/Beat_smplx/"

    # source_BEAT_path = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1"
    # sample_audio_feat, sample_text_feat, text, audio_tag = read_data(sample_audio_files, sample_text_files, segment_length, BEAT_wav_feat, HUMANML3D_text_feat, save_path, GT_HUMANML3D_motion)

    sample_audio_feat, sample_text_feat, text, audio_tag = None, None, None, None

    if version == 'v0_0' or version == 'v0_1':
        n_seed = 10
    elif version == 'v0_2' or version == 'v0_3' or version == 'v0_4' or version == "v1_2" or version == "v1_3" or version == "v1_4":
        n_seed = 20
    elif version == "v0_2_1" or version == "v0_3_1":
        n_seed = 40
    elif version == 'v3_6':
        n_seed = 30
    elif version == 'v2_0' or version == 'v2_5' or version == 'v2_1' or version == 'v2_6' or version == 'ref'\
            or version == 'v3_2' or version == 'v3_3' or version == 'v3_4' or version == 'v3_5'\
            or version == 'v3_0' or version == 'v3_1' :
        n_seed = 0
    else:
        raise ValueError("version not found!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    process(sample_audio_feat, sample_text_feat, segment_length, save_path, model_name, n_seed=n_seed, version=version, text=text)
