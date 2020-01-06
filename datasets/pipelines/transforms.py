import random
import cv2 as cv
import numpy as np
from ..registry import PIPELINES


'''
sitk image: (width, height, depth)
sitk ndarray: (depth, height, width)
numpy ndarray: (height, width, channel)
torch tensor shape: (channel, height, width)
'''


def pad2d(data, shape, fill_value):
    padded_data = np.full(shape, fill_value, dtype=data.dtype)
    padded_data[:data.shape[0], :data.shape[1]] = data
    return padded_data


@PIPELINES.register_module
class TargetFromBoxes(object):

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, results):
        input_data = results['input']
        target_data = input_data.copy()

        if 'gt_boxes' in results:
            boxes = results['gt_boxes']
            for x1, y1, x2, y2 in boxes:
                input_data[y1:y2 + 1, x1:x2 + 1] = self.fill_value

        results['input'] = input_data
        results['target'] = target_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(fill_value={})'.format(self.fill_value)


@PIPELINES.register_module
class TargetFromRepair(object):

    def __init__(self, block_range=(16, 32), fill_value=0):
        self.block_range = block_range
        self.fill_value = fill_value

    def __call__(self, results):
        input_data = results['input']
        target_data = input_data.copy()

        y = input_data.shape[0] // 2
        x = input_data.shape[1] // 2
        y_r = np.random.randint(*self.block_range) // 2
        x_r = np.random.randint(*self.block_range) // 2
        input_data[y - y_r:y + y_r + 1, x - x_r:x + x_r + 1] = self.fill_value

        results['input'] = input_data
        results['target'] = target_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(block_range={}, fill_value={})'.format(self.block_range, self.fill_value)


@PIPELINES.register_module
class TargetFromMotion(object):

    def __init__(self, invariant_prob=0.1, degree=(10, 20)):
        self.invariant_prob = invariant_prob
        self.degree = degree

    def __call__(self, results):
        input_data = results['input']
        target_data = input_data.copy()

        if random.random() >= self.invariant_prob:
            angle = np.random.randint(0, 360)
            degree = np.random.randint(*self.degree)

            # 模糊kernel，degree越大，越模糊
            kernel = np.diag(np.ones(degree))
            M = cv.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            kernel = cv.warpAffine(kernel, M, (degree, degree))
            kernel = kernel / degree

            input_data = cv.filter2D(input_data, -1, kernel)
            cv.normalize(input_data, input_data, target_data.min(), target_data.max(), cv.NORM_MINMAX)

        results['input'] = input_data
        results['target'] = target_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(invariant_prob={}, degree={})'.format(self.invariant_prob, self.degree)


@PIPELINES.register_module
class NormalizeCustomize(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        input_data = results['input']
        a, b = input_data.min(), input_data.max()

        mean, std = (a + b) / 2, (b - a) / 2
        input_data = (input_data - mean) / (std + self.eps)

        results['input'] = input_data
        results['norm_cfg'] = dict(mean=mean, std=std)

        if 'target' in results:
            target_data = results['target']
            a, b = target_data.min(), target_data.max()

            mean, std = (a + b) / 2, (b - a) / 2
            target_data = (target_data - mean) / (std + self.eps)

            results['target'] = target_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class NormalizeInstance(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        input_data = results['input']
        mean, std = input_data.mean(), input_data.std()
        input_data = (input_data - mean) / (std + self.eps)

        results['input'] = input_data
        results['norm_cfg'] = dict(mean=mean, std=std)

        if 'target' in results:
            target_data = results['target']
            mean, std = target_data.mean(), target_data.std()
            target_data = (target_data - mean) / (std + self.eps)

            results['target'] = target_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class RandomCrop(object):

    def __init__(self, crop_size, clean=True):
        self.crop_size = crop_size
        self.clean = clean

    def __call__(self, results):
        input_data = results['input']
        assert 0 < self.crop_size <= min(input_data.shape)

        for i in range(30):
            y = np.random.randint(0, input_data.shape[0] - self.crop_size + 1)
            x = np.random.randint(0, input_data.shape[1] - self.crop_size + 1)
            patch = np.array([x, y, x + self.crop_size, y + self.crop_size])

            # adjust boxes
            if 'gt_boxes' in results:
                boxes = results['gt_boxes']
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                valid_inds = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                if self.clean == np.any(valid_inds):
                    continue

                results['gt_boxes'] = boxes[valid_inds, :]

            # adjust masks
            if 'gt_masks' in results:
                valid_masks = [mask[patch[1]:patch[3], patch[0]:patch[2]] for mask in results['gt_masks']]
                valid_masks = [mask for mask in valid_masks if mask.max() > 0]
                results['gt_masks'] = valid_masks

            # adjust target
            if 'target' in results:
                target_data = results['target'][patch[1]:patch[3], patch[0]:patch[2]]
                results['target'] = target_data

            input_data = input_data[patch[1]:patch[3], patch[0]:patch[2]]

            results['input'] = input_data
            results['ori_shape'] = input_data.shape

            return results
        return None

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={}, to_clear={})'.format(self.crop_size, self.clean)


@PIPELINES.register_module
class Pad(object):

    def __init__(self, size_divisor=32, fill_value=0):
        self.size_divisor = size_divisor
        self.fill_value = fill_value

    def __call__(self, results):
        input_data = results['input']

        pad_shape = tuple(int(np.ceil(v / self.size_divisor)) * self.size_divisor for v in input_data.shape)

        padded_data = pad2d(input_data, pad_shape, self.fill_value)
        results['input'] = padded_data
        results['pad_shape'] = padded_data.shape

        # adjust masks
        if 'gt_masks' in results:
            padded_masks = [pad2d(mask, pad_shape, 0) for mask in results['gt_masks']]
            results['gt_masks'] = np.stack(padded_masks, axis=0)

        # adjust target
        if 'target' in results:
            padded_data = pad2d(results['target'], pad_shape, self.fill_value)
            results['target'] = padded_data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(size_divisor={}, fill_value={})'.format(self.size_divisor, self.fill_value)
