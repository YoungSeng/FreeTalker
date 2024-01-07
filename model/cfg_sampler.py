import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        # assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        # self.rot2xyz = self.model.rot2xyz
        # self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        y_cond_ = deepcopy(y)
        # y_cond_['uncond'] = True
        # y_cond_['audio'] = False

        y_cond_['audio'] = y_cond_['audio_']        # w/o audio


        out = self.model(x, timesteps, y)
        out_cond_ = self.model(x, timesteps, y_cond_)
        return out_cond_ + (y['scale'].view(-1, 1, 1, 1) * (out - out_cond_))

