from model.mdm import MDM
# from diffusion import gaussian_diffusion as gd
# from diffusion.respace import SpacedDiffusion, space_timesteps

from diffusion_priormdm import gaussian_diffusion as gd
from diffusion_priormdm.respace import SpacedDiffusion, space_timesteps


from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print(f"Missing keys: {missing_keys}", f"Unexpected keys: {unexpected_keys}")
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, model_version, device, sample_mode=False):
    # n_seed
    if model_version == 'v0_1':
        n_seed = 10
    elif model_version == 'v0_2' or model_version == 'v0_3' or model_version == 'v0_4' or model_version == 'v0_5' or model_version == 'v1_2' or model_version == 'v1_3' or model_version == 'v1_4':
        n_seed = 20
    elif model_version == 'v0_2_1' or model_version == 'v0_3_1':
        n_seed = 40
    elif model_version == 'v2_2' or model_version == 'v2_3' or model_version == 'v2_7' or model_version == 'v2_8' \
            or model_version == 'v3_6' or model_version == 'v3_7':
        n_seed = 30
    elif model_version == 'v2_0' or model_version == 'v2_5' or model_version == 'v2_1' or model_version == 'v2_6' \
            or model_version == 'ref' or model_version == 'v3_0' or model_version == 'v3_5' or model_version == 'v3_1'\
            or model_version == 'v3_2' or model_version == 'v3_3' or model_version == 'v3_4':
        n_seed = 0
    else:
        raise ValueError('invalid model_version: {}'.format(model_version))
    # n_joints
    if 'v0' in model_version:
        n_joints = 168
        latent_dim = 256
    elif 'v1' in model_version:
        n_joints = 1523
        latent_dim = 256
    elif 'v2' in model_version or 'v3' in model_version:
        n_joints = 659
        if 'v3_2' not in model_version and 'v3_3' not in model_version and 'v3_4' not in model_version \
                and 'v3_6' not in model_version and 'v3_7' not in model_version:
            latent_dim = 256
        else:
            latent_dim = 512
    elif 'ref' in model_version:
        n_joints = 263
        latent_dim = 512
    else:
        raise ValueError('invalid model_version: {}'.format(model_version))
    print('n_joints: {}, latent_dim: {}'.format(n_joints, latent_dim))
    model = MDM(modeltype='', njoints=n_joints, nfeats=1, num_actions=None, cond_mode='text_audio',      ## 'text' njoints 168
             latent_dim=latent_dim, num_layers=8, num_heads=4, dropout=0.1, clip_version='ViT-B/32',
             dataset='humanml', model_version=model_version, n_seed=n_seed, device=device, batch_size=args.batch_size,
                n_frames=args.n_frames, split_para=args.split_para, sample_mode=sample_mode)
    diffusion = create_gaussian_diffusion(args, device)
    return model, diffusion


def create_gaussian_diffusion(args, device):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )