import re
import torch
from torch.utils.data import DataLoader
from torch._six import container_abcs, string_classes, int_classes


np_str_obj_array_pattern = re.compile(r'[SaUO]')
error_msg_fmt = 'batch must contain tensors, numbers, dicts or lists; found {}'


def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            raise TypeError(error_msg_fmt.format(elem.dtype))
        return default_collate([torch.from_numpy(b) for b in batch])
    elif isinstance(elem, float):
        return torch.tensor(batch)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        if elem.get('cpu_only', False):
            return {key: [d[key] for d in batch] for key in elem}
        else:
            return {key: default_collate([d[key] for d in batch]) for key in elem}

    raise TypeError(error_msg_fmt.format(type(elem)))


def build_dataloader(dataset, imgs_per_gpu, workers_per_gpu, num_gpus=1, dist=True, **kwargs):
    shuffle = kwargs.get('shuffle', True)

    if dist:
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=default_collate,
                             pin_memory=False,
                             **kwargs)

    return data_loader
