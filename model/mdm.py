import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import sys
[sys.path.append(i) for i in ['..']]
from model.rotation2xyz import Rotation2xyz
import pdb


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, model_version='v0_0', n_seed=10, device='cuda：0',
                 batch_size=-1, n_frames=-1, split_para=1, sample_mode=False, **kargs):
        super().__init__()

        self.batch_size = batch_size
        self.n_frames = n_frames
        self.split_para = split_para
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.device = device
        self.sample = sample_mode

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.model_version = model_version
        print('model version', model_version)
        self.n_seed = n_seed
        if model_version == 'v0_0':
            self.source_audio_dim = 1133
            self.embed_seed = nn.Linear(self.njoints, self.latent_dim)
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim // 2)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim // 2)
            self.input_process2 = nn.Linear(self.latent_dim * 3, self.latent_dim)
            # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        elif model_version == 'v0_1' or model_version == 'v0_2' or model_version == 'v1_2' or model_version == 'v2_2':
            self.source_audio_dim = 1133
            self.embed_seed = nn.Linear(self.njoints, self.latent_dim)
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 3, self.latent_dim)

        elif model_version == 'v0_3' or model_version == 'v1_3'  or model_version == 'v2_3'\
                or model_version == 'v0_5' or model_version == 'v3_6':
            self.source_audio_dim = 1133
            self.embed_seed = nn.Linear(self.njoints, self.latent_dim)
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)

            if model_version == 'v0_5':
                from process.process_SMPLX import smplx_model_init, smplx_frame_by_frame_mdm, smplx_frame_by_frame_mdm_2
                self.smplx_model = smplx_model_init("/apdcephfs/private_yyyyyyyang/code/human_body_prior/support_data/dowloads/models", batch_size=self.batch_size * self.n_frames // self.split_para).to(self.device).eval()
                for params in self.smplx_model.parameters():
                    params.requires_grad = False
                self.smplx_frame_by_frame = smplx_frame_by_frame_mdm_2

        elif model_version == 'v0_4' or model_version == 'v1_4' or model_version == 'v2_4':
            '''
            pip install einops
            '''
            import sys
            [sys.path.append(i) for i in ['./model']]
            from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
            from local_attention import LocalAttention

            self.source_audio_dim = 1133
            self.embed_seed = nn.Linear(self.njoints, self.latent_dim)
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 3, self.latent_dim)
            self.num_head = 8
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.cross_local_attention = LocalAttention(
                dim=48,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=10,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,
                # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )


        elif model_version == 'v2_0' or model_version == 'v2_1' or model_version == 'v3_0' or model_version == 'v3_1' or model_version == 'v3_2':
            self.source_audio_dim = 1133
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)

        elif model_version == 'v2_5' or model_version == 'v2_6' or model_version == 'v3_5' or model_version == 'v3_3' or model_version == 'v3_4':
            import sys
            [sys.path.append(i) for i in ['./model']]
            from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
            from local_attention import LocalAttention

            self.source_audio_dim = 1133
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
            self.num_head = 8
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.cross_local_attention = LocalAttention(
                dim=64,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=10,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,
                # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )

        elif model_version == 'v2_7' or model_version == 'v2_8' or model_version == 'v3_7':
            import sys
            [sys.path.append(i) for i in ['./model']]
            from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
            from local_attention import LocalAttention

            self.source_audio_dim = 1133
            self.embed_seed = nn.Linear(self.njoints, self.latent_dim)
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.embed_audio = nn.Linear(self.source_audio_dim, self.latent_dim)
            self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
            self.num_head = 8
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.cross_local_attention = LocalAttention(
                dim=48,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=10,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,
                # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )

        elif model_version == 'ref':
            self.clip_dim = 512
            clip_version = 'ViT-B/32'

            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            print('EMBED TEXT')
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        # clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
        #                                         jit=False)  # Must set jit=False for training
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False, download_root=str("/apdcephfs/private_yyyyyyyang/clip/"))  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):

        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
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
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape


        force_mask = y.get('uncond', False)

        if self.model_version == 'v0_0':
            emb = self.embed_timestep(timesteps).repeat(nframes, 1, 1)      # [frames, bs, d]
            embed_seed = self.embed_seed(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)        # (n_seed, bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)     # (frames-n_seed, bs, d/2)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask))      # (bs, d/2)

            text_frames_emb = embed_text.repeat(nframes - self.n_seed, 1, 1)     # (frames-n_seed, bs, d/2)
            audio_text_frames_emb = torch.cat((embed_audio, text_frames_emb), axis=2)     # (frames-n_seed, bs, d)
            motion_emb = torch.cat((embed_seed, audio_text_frames_emb), axis=0)     # (frames, bs, d)

            x = self.input_process(x)       # (frames, bs, d)
            xseq = torch.cat((emb, motion_emb, x), axis=2)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)

            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output


        elif self.model_version == 'v0_1' or self.model_version == 'v0_2' or self.model_version == 'v1_2' or self.model_version == 'v2_2':
            emb = self.embed_timestep(timesteps).repeat(nframes, 1, 1)      # [frames, bs, d]
            embed_seed = self.embed_seed(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (n_seed, bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)  # (frames-n_seed, bs, d)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask))  # (bs, d)
            text_frames_emb = embed_text.repeat(nframes, 1, 1)  # (frames, bs, d)
            emb += text_frames_emb  # (frames, bs, d)

            seed_audio = torch.cat((embed_seed, embed_audio), axis=0)  # (frames, bs, d)
            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((emb, seed_audio, x), axis=2)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)

            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == 'v0_3' or self.model_version == 'v1_3' or self.model_version == 'v2_3'\
                or self.model_version == 'v0_5' or self.model_version == 'v3_6':
            emb = self.embed_timestep(timesteps)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask)).unsqueeze(0)
            emb = torch.cat((embed_text, emb), axis=2)  # (bs, d)
            embed_seed = self.embed_seed(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (n_seed, bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)  # (frames-n_seed, bs, d)
            seed_audio = torch.cat((embed_seed, embed_audio), axis=0)  # (frames, bs, d)
            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((seed_audio, x), axis=2)
            xseq = torch.cat((emb, xseq), axis=0)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == 'v0_4' or self.model_version == 'v1_4' or self.model_version == 'v2_4':
            emb = self.embed_timestep(timesteps)
            embed_seed = self.embed_seed(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (n_seed, bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)  # (frames-n_seed, bs, d)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask))  # (bs, d)
            emb = emb + embed_text.unsqueeze(0)  # (bs, d)

            text_frames_emb = emb.repeat(nframes, 1, 1)  # (frames, bs, d)

            seed_audio = torch.cat((embed_seed, embed_audio), axis=0)  # (frames, bs, d)
            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((text_frames_emb, seed_audio, x), axis=2)
            xseq = self.input_process2(xseq)

            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = xseq.permute(1, 0, 2)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)
            xseq, _ = self.apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=y['mask'])
            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            xseq = torch.cat((emb, xseq), axis=0)
            xseq = xseq.permute(1, 0, 2)
            xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
            pos_emb = self.rel_pos(xseq)
            xseq, _ = self.apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
            xseq = xseq_rpe.permute(0, 2, 1, 3)
            xseq = xseq.view(bs, nframes + 1, -1)
            xseq = xseq.permute(1, 0, 2)
            output = self.seqTransEncoder(xseq)[1:]

            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == 'v2_0' or self.model_version == 'v2_1' or self.model_version == 'v3_0' \
                or self.model_version == 'v3_1' or self.model_version == 'v3_2':
            emb = self.embed_timestep(timesteps)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask)).unsqueeze(0)
            emb = torch.cat((embed_text, emb), axis=2)  # (bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)  # (frames-n_seed, bs, d)
            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((embed_audio, x), axis=2)
            xseq = torch.cat((emb, xseq), axis=0)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == 'v2_5' or self.model_version == 'v2_6' or self.model_version == 'v3_5' \
                or self.model_version == 'v3_3' or self.model_version == 'v3_4':
            emb = self.embed_timestep(timesteps)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask)).unsqueeze(0)
            emb = torch.cat((embed_text, emb), axis=2)  # (bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)

            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = embed_audio.permute(1, 0, 2)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)
            xseq, _ = self.apply_rotary_pos_emb(xseq, xseq, pos_emb)

            if self.sample:
                mask = torch.ones((bs, 1, 1, xseq.shape[1])).bool().to(xseq.device)
            else:
                mask = y['mask']
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=mask)
            # [24, 160, 32], [3, 8], [3, 1, 1, 160]
            # [16, 260, 32], [2, 8], [3, 1, 1, 160]
            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((xseq, x), axis=2)
            xseq = torch.cat((emb, xseq), axis=0)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output


        elif self.model_version == 'v2_7' or self.model_version == 'v2_8' or self.model_version == 'v3_7':

            emb = self.embed_timestep(timesteps)
            embed_text = self.embed_text(self.mask_cond(y['text'], force_mask=force_mask)).unsqueeze(0)
            emb = torch.cat((embed_text, emb), axis=2)  # (bs, d)
            embed_seed = self.embed_seed(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (n_seed, bs, d)
            embed_audio = self.embed_audio(y['audio']).permute(1, 0, 2)  # (frames-n_seed, bs, d)
            xseq = torch.cat((embed_seed, embed_audio), axis=0)  # (frames, bs, d)

            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = xseq.permute(1, 0, 2)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)
            xseq, _ = self.apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=y['mask'])
            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            x = self.input_process(x)  # (frames, bs, d)
            xseq = torch.cat((xseq, x), axis=2)
            xseq = torch.cat((emb, xseq), axis=0)
            xseq = self.input_process2(xseq)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == 'ref':
            """
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            """
            bs, njoints, nfeats, nframes = x.shape
            emb = self.embed_timestep(timesteps)  # [1, bs, d]

            force_mask = y.get('uncond', False)
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

            x = self.input_process(x)

            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            return output

    # def _apply(self, fn):
    #     super()._apply(fn)
    #     self.rot2xyz.smpl_model._apply(fn)


    # def train(self, *args, **kwargs):
    #     super().train(*args, **kwargs)
    #     self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


if __name__ == '__main__':
    '''
    python -m model.mdm
    '''

    njoints = 659
    bs = 2
    n_frames = 180      # 20fps, 10frames predict 8s
    n_seed = 0
    audio_feature_dim = 1133
    text_feature_dim = 512

    device = torch.device('cpu')       # cpu or cuda:0
    MDM_model = MDM(modeltype='', njoints=njoints, nfeats=1, num_actions=None, cond_mode='text_audio',      ## 'text'
             latent_dim=512, num_layers=8, num_heads=4, dropout=0.1, clip_version='ViT-B/32',
             dataset='humanml', model_version='v3_4', n_seed=n_seed).to(device)

    x = torch.randn(bs, njoints, 1, n_frames).to(device)
    t = torch.tensor([12, 85]).to(device)
    model_kwargs_ = {'y': {}}
    # model_kwargs_['y']['text'] = ['unconstrained', 'unconstrained']
    model_kwargs_['y']['text'] = torch.randn(bs, text_feature_dim).to(device)       # ['a man lifts something on his left and places it down on his right.', 'a person jumps sideways to the left']
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(device)  # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed, audio_feature_dim)       # attention4
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames, audio_feature_dim)       # v2_0

    # model_kwargs_['y']['style'] = torch.randn(bs, style_dim)
    # model_kwargs_['y']['mask_local'] = torch.ones(bs, n_frames).bool()
    # model_kwargs_['y']['seed'] = x[..., 0:n_seed]  # attention3/4
    # model_kwargs_['y']['seed_last'] = x[..., -n_seed:]  # attention5
    # model_kwargs_['y']['gesture'] = torch.randn(bs, n_frames, njoints)
    y = MDM_model(x, t, model_kwargs_['y'])  # [bs, njoints, nfeats, nframes]
    print(y.shape)
