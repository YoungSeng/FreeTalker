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


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating data loader...")
    dataset = MotionDataset(h5file=args.h5file_path,
                            statistics_path=args.statistics_path,
                            version='v3')

    data = DataLoader(dataset, num_workers=4,
                              sampler=RandomSampler(len(dataset), dataset.dataset_count),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=t2m_collate)

    print("creating model and diffusion...")
    # model_version = 'v0_2'
    model_version = args.save_dir.replace('save/my_', '')

    model, diffusion = create_model_and_diffusion(args, model_version=model_version, device=device)
    model.to(device)
    # model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, model, diffusion, data).run_loop()


if __name__ == "__main__":
    '''
    python -m train.train_mdm --save_dir save/my_v0_5 --overwrite --batch_size 256 --n_frames 190 --split_para 32 --lambda_rcxyz 1 --lambda_vel_rcxyz 1 --lambda_vel 1
    python -m train.train_mdm --save_dir save/my_v1_2 --overwrite --batch_size 256 --n_frames 190 --n_seed 20
    python -m train.train_mdm --save_dir save/my_v2_1 --overwrite --batch_size 256 --n_frames 160 --n_seed 0
    python -m train.train_mdm --save_dir save/my_v3_0 --overwrite --batch_size 256 --n_frames 180 --n_seed 0
    '''
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.is_available())
    print('number of GPUs', torch.cuda.device_count())
    main()
