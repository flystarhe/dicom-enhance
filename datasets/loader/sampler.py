import math
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler as _Sampler


def get_dist_info():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super(DistributedSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(_Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += math.ceil(size / self.samples_per_gpu) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue

            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)

            num_extra = math.ceil(size / self.samples_per_gpu) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
                   for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(_Sampler):

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        assert hasattr(dataset, 'flag')
        _rank, _num_replicas = get_dist_info()

        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += math.ceil(size / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(size, generator=g))].tolist()

                extra = math.ceil(size / self.samples_per_gpu / self.num_replicas) * \
                    self.samples_per_gpu * self.num_replicas - len(indice)
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)
        assert len(indices) == self.total_size

        indices = [indices[j] for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator=g))
                   for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
