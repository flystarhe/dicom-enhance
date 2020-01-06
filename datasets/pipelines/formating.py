import torch
import numpy as np
from ..registry import PIPELINES


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.tensor([data])
    elif isinstance(data, float):
        return torch.tensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(type(data)))


@PIPELINES.register_module
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class SliceToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key][None, ...])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class Collect(object):

    def __init__(self, keys, meta_keys=('dicom', 'ori_shape', 'norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data_meta = {}
        for key in self.meta_keys:
            data_meta[key] = results[key]
        data_meta.setdefault('cpu_only', True)

        data = dict(meta=data_meta)
        for key in self.keys:
            data[key] = results[key]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(self.keys, self.meta_keys)
