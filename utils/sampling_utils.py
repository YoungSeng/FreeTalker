from copy import deepcopy
import torch
from utils import dist_util
import pdb
import numpy as np
import os


def unfold_sample_arb_len(sample, handshake_size, step_sizes, final_n_frames, model_kwargs):
    old_sample = deepcopy(sample)
    new_shape = list(old_sample.shape)
    new_shape[0] = 1
    new_shape[-1] = final_n_frames
    sample = torch.zeros(new_shape, dtype=sample.dtype, device=sample.device)
    sample[0, :, :, :model_kwargs['y']['lengths'][0]] = old_sample[0, :, :, :model_kwargs['y']['lengths'][0]]
    for sample_i, len_i in enumerate(step_sizes):
        if sample_i == 0:
            continue
        start = step_sizes[sample_i-1]
        sample[0, :, :, start:len_i] = old_sample[sample_i, :, :, handshake_size:model_kwargs['y']['lengths'][sample_i]]
    return sample


def double_take_arb_len(args, diffusion, model, model_kwargs, n_frames, eval_mode=False, guidacnce_param=1.0):
    # FIXME - not working for num_repetitions > 1
    debug = False       # args.debug_double_take
    blend_len = 10    # args.blend_len
    sample_fn = diffusion.p_sample_loop
    handshake_size = 20         # args.handshake_size

    samples_per_rep_list = []
    samples_type = []
    orig_sample = []
    batch_size = len(model_kwargs['y']['text'])

    transition = torch.zeros(n_frames)
    transition[:handshake_size] = 1.  #[T0 T0 M1 M1 M1 M1 M1 M1 T1 T1] Motion sanwitch
    transition = torch.tile(transition.unsqueeze(0), dims=(batch_size, 1))
    transition[0, :handshake_size] = 0
    for ii in range(batch_size - 1):
        transition[ii,
        model_kwargs['y']['lengths'][ii] - handshake_size: model_kwargs['y']['lengths'][ii]] = 1.0
    model_kwargs['y']['is_transition'] = transition

    # Unfolding - orig
    sample = sample_fn(
        model,
        (batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0 if not debug else 998,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=not eval_mode,
        dump_steps=None,
        noise=None,
        const_noise=False,
        unfolding_handshake=handshake_size,
        arb_len=True,
        second_take_only=False,      # args.second_take_only
    )

    # print(sample.shape)
    orig_sample.append(sample)

    # save_path = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000"
    # orig_sample = []
    # for i in range(3):
    #     tmp = np.load(os.path.join(save_path, "positions_real_{}_.npy".format(i)))
    #     mean = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/" + "v3_mean.npy")
    #     std = np.load("/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/" + "v3_std.npy")
    #     tmp = (tmp - mean) / std
    #
    #     orig_sample.append(torch.tensor(tmp, dtype=torch.float32, device=sample.device))
    #
    # sample = torch.stack(orig_sample).permute(0, 2, 1).unsqueeze(2)
    # print(sample.shape)

    if True:        # args.double_take
        old_guidacnce_param = guidacnce_param     # 2.5           # args.guidance_param
        guidance_param = 0.  # Force unconditioned generation
        model_kwargs['y']['scale'] = torch.ones(batch_size-1, device=dist_util.dev()) * guidance_param

        new_sample_seq_len = (sample.shape[-1] - 2 * handshake_size) * 2 + handshake_size

        bs, feats, joints, seq_len = sample.shape
        new_sample = torch.zeros((bs-1, feats, joints, new_sample_seq_len), dtype=sample.dtype, device=sample.device)

        generated_motion = []
        right_constraint = []
        left_constraint = []

        for ii in range(bs):
            generated_motion.append(deepcopy(sample[ii, :, :, handshake_size: model_kwargs['y']['lengths'][ii]-handshake_size])) # w/o start and end
            left_constraint.append(deepcopy(sample[ii, :, :, :handshake_size]))  # left side
            right_constraint.append(deepcopy(sample[ii, :, :, model_kwargs['y']['lengths'][ii] - handshake_size: model_kwargs['y']['lengths'][ii]]))

        buffer = []
        for ii in range(bs):
            buffer.append(int(model_kwargs['y']['lengths'][ii]) - 2*handshake_size)
        for ii in range(bs - 1):  # run over bs
            new_sample[ii, :, :, :buffer[ii]] = generated_motion[ii]
            new_sample[ii, :, :, buffer[ii]: buffer[ii]+handshake_size] = right_constraint[ii] # add transition
            new_sample[ii, :, :, buffer[ii]+handshake_size : buffer[ii]+handshake_size+buffer[ii+1]] = generated_motion[ii + 1]

        # "in between"
        model_kwargs['y']['inpainted_motion'] = new_sample
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(new_sample, dtype=torch.float,
                                                               device=new_sample.device)

        for ii in range(bs - 1):  # run over bs
            # model_kwargs['y']['inpainting_mask'][ii, :, :, buffer[ii]: buffer[ii]+handshake_size] = 0.3
            if blend_len >= 2:
                arange_param = 0.85
                model_kwargs['y']['inpainting_mask'][ii, :, :, buffer[ii] - blend_len: buffer[ii]] = \
                    torch.arange(arange_param, 0.0, -arange_param / int(blend_len))
                model_kwargs['y']['inpainting_mask'][ii, :, :, buffer[ii] + handshake_size: buffer[ii] + handshake_size + blend_len] = \
                    torch.arange(0.0, arange_param, arange_param / int(blend_len))

        transition_orig = deepcopy(model_kwargs['y']['is_transition'])
        transition = torch.zeros(new_sample_seq_len)
        transition = torch.tile(transition.unsqueeze(0), dims=(bs-1, 1))
        model_kwargs['y']['is_transition'] = transition
        model_kwargs['y']['uncond'] = 1.0
        last_text = model_kwargs['y']['text'][-1]
        model_kwargs['y']['text'] = model_kwargs['y']['text'][:bs-1]
        sample_fn = diffusion.p_sample_loop  # double take sample function
        n_frames = new_sample_seq_len
        orig_lens = deepcopy(model_kwargs['y']['lengths'])
        for ii in range (len(model_kwargs['y']['lengths'])-1):
            model_kwargs['y']['lengths'][ii] = model_kwargs['y']['lengths'][ii] + model_kwargs['y']['lengths'][ii+1] - 3*handshake_size
        model_kwargs['y']['lengths'] = model_kwargs['y']['lengths'][:-1]

        orig_audio = deepcopy(model_kwargs['y']['audio'])
        list_audio = []
        for ii in range (len(model_kwargs['y']['audio'])-1):
            list_audio.append(torch.cat((model_kwargs['y']['audio'][ii], model_kwargs['y']['audio'][ii+1][3*handshake_size:]), dim=0))

        list_audio_ = []
        for ii in range (len(model_kwargs['y']['audio_'])-1):
            list_audio_.append(torch.cat((model_kwargs['y']['audio_'][ii], model_kwargs['y']['audio_'][ii+1][3*handshake_size:]), dim=0))

        list_audio = torch.stack(list_audio)
        model_kwargs['y']['audio'] = list_audio

        list_audio_ = torch.stack(list_audio_)
        model_kwargs['y']['audio_'] = list_audio_

        skip_steps_double_take = 100      # 100

        double_take_sample = sample_fn(
            model,
            (batch_size-1, model.njoints, model.nfeats, n_frames),      # (2, 659, 1, 260)
            clip_denoised=False,
            model_kwargs=model_kwargs,
            # audio [2, 260, 1133]
            # text [2, 512]
            skip_timesteps=skip_steps_double_take if not debug else 998,  # 0 is the default value - i.e. don't skip any step
            init_image= new_sample, #TODO!! check if plausible or not!      # [2, 659, 1, 260]
            progress=not eval_mode,     # False
            dump_steps=None,
            noise=None,
            const_noise=False,
            repaint_samples=1,
            unfolding_handshake=0,
            arb_len = False
        )
        model_kwargs['y']['lengths'] = orig_lens
        # rebuild_orig:
        rebuild_sample = torch.zeros_like(sample)

        transitions, right_side, left_side = [], [], []
        for ii in range(bs - 1):  # run over bs
            transitions.append(double_take_sample[ii, :, :, buffer[ii]: buffer[ii]+handshake_size])
            right_side.append(double_take_sample[ii, :, :, buffer[ii] + handshake_size: buffer[ii] + handshake_size + blend_len]) # M1 blending..
            left_side.append(double_take_sample[ii, :, :, buffer[ii] - blend_len:buffer[ii]]) # M0 blending...


        rebuild_sample[0, :, :, :handshake_size] = left_constraint[0] # Fill missing
        rebuild_sample[-1, :, :, buffer[-1]+handshake_size: buffer[-1]+2*handshake_size] = right_constraint[-1] # Fill missing

        for ii in range(bs - 1):
            rebuild_sample[ii + 1, :, :, :handshake_size] = transitions[ii]
            rebuild_sample[ii, :, :, handshake_size: buffer[ii]+handshake_size] = generated_motion[ii]
            rebuild_sample[ii, :, :, buffer[ii]+handshake_size: buffer[ii]+2*handshake_size] = transitions[ii]
            rebuild_sample[ii, :, :, handshake_size + buffer[ii]-blend_len: handshake_size + buffer[ii]] = left_side[ii]
            # if ii > 0:
        rebuild_sample[-1, :, :, handshake_size: buffer[-1] + handshake_size] = generated_motion[-1]
        for ii in range(bs - 1):
            rebuild_sample[ii+1, :, :, handshake_size:handshake_size + blend_len] = right_side[ii]

        double_take_sample = deepcopy(rebuild_sample)
        samples_per_rep_list.append(double_take_sample)
        samples_type.append('')

        args.guidance_param = old_guidacnce_param
        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * args.guidance_param


        model_kwargs['y'].pop('inpainted_motion')
        model_kwargs['y'].pop('inpainting_mask')
        model_kwargs['y'].pop('uncond')
        model_kwargs['y']['is_transition'] = deepcopy(transition_orig)
        try:
            model_kwargs['y']['text'].append(last_text)
        except:
            # 将tensor2增加一个维度以匹配tensor1的形状
            last_text = last_text.unsqueeze(0)
            # 沿着第0个维度拼接两个张量
            model_kwargs['y']['text'] = torch.cat((model_kwargs['y']['text'], last_text), dim=0)

    return samples_per_rep_list, samples_type, orig_sample

