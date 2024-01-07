import pdb

import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import os


class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, statistics_path, version='v0'):

        if 'val' in h5file.split("/")[-1].split(".")[0]:
            print("Loading validation data...")
            self.mode = 'val'
        elif 'train' in h5file.split("/")[-1].split(".")[0]:
            print("Loading training data...")
            self.mode = 'train'
        self.h5 = h5py.File(h5file, "r")
        # self.len_h5 = 18
        self.len_h5 = len(self.h5.keys())
        self.mean = np.load(os.path.join(statistics_path, version + "_mean.npy"))
        self.std = np.load(os.path.join(statistics_path, version + "_std.npy"))
        self.dataset = [self.h5[str(i)]["dataset"][:][0] for i in range(self.len_h5)]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(self.len_h5)]
        self.text = [self.h5[str(i)]["text"][:] for i in range(self.len_h5)]
        self.motion = [(self.h5[str(i)]["motion"][:] - self.mean) / self.std for i in range(self.len_h5)]
        if self.mode == 'val':
            self.name = [self.h5[str(i)]["name"][:][0] for i in range(self.len_h5)]
        self.h5.close()
        print("Total clips:", len(self.motion))
        self.segment_length = 180       # 160 frames = 8s， v2

        self.dataset_count = np.array([self.dataset.count(b'BEAT'), self.dataset.count(b'HUMANML3D')])




    def __len__(self):
        return self.len_h5

    def __getitem__(self, idx):
        segment_len = self.audio[idx].shape[0]
        if self.mode == 'val':
            start_frame = 0
        else:
            start_frame = np.random.randint(0, max(1, segment_len - self.segment_length))
        motion = self.motion[idx][start_frame:start_frame + self.segment_length]
        audio = self.audio[idx][start_frame:start_frame + self.segment_length]
        text = torch.FloatTensor(self.text[idx])

        m_length = motion.shape[0]
        if m_length < self.segment_length:
            motion = np.concatenate([motion, np.zeros((self.segment_length - m_length, motion.shape[1]))], axis=0)
            audio = np.concatenate([audio, np.zeros((self.segment_length - m_length, audio.shape[1]))], axis=0)

        from_data = self.dataset[idx]

        if self.mode == 'val':
            return motion, audio, text, m_length, self.name[idx]
        else:
            return motion, audio, text, m_length, from_data


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    textbatch = collate_tensors([b['text'] for b in notnone_batches])
    cond['y'].update({'text': textbatch})

    audiobatch = collate_tensors([b['audio'] for b in notnone_batches])
    cond['y'].update({'audio': audiobatch})

    from_data = [b['from_data'] for b in notnone_batches]
    cond['y'].update({'from_data': from_data})

    return motion, cond


def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1),
        'audio': torch.FloatTensor(b[1]),
        'text': b[2],
        'lengths': b[3],
        'from_data': b[4],
    } for b in batch]
    return collate(adapted_batch)


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, n_segments, split_list):
        self.n_segments = n_segments
        dataset_a_size = split_list[0]
        dataset_b_size = split_list[1]
        # 计算每个数据集中的元素权重
        print(dataset_a_size, dataset_b_size)
        weights = [1.0 / dataset_a_size] * dataset_a_size + [1.0 / dataset_b_size] * dataset_b_size
        # 对权重进行归一化
        weights /= np.array(weights).sum()

        self.weights = weights

    def __iter__(self):     # 根据权重随机采样
        while True:
            yield np.random.choice(self.n_segments, p=self.weights)


if __name__ == '__main__':
    '''
    python -m data_loaders.h5_data_loader
    '''
    print("Loading dataset into memory ...")
    dataset = MotionDataset(h5file="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/v2_val.h5",
                            statistics_path="/apdcephfs/private_yyyyyyyang/code/mdm/prcocessed_data/",
                            version="v2")

    train_loader = DataLoader(dataset, num_workers=4,
                              sampler=RandomSampler(len(dataset), dataset.dataset_count),
                              batch_size=4,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=t2m_collate)

    for batch_i, batch in enumerate(train_loader, 0):
        motion, cond = batch     # (128, 150, 1435), (128, 150, 744), (128, 17)
        print(batch_i)
        pdb.set_trace()
