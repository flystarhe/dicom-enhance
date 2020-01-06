import os.path as osp
import SimpleITK as sitk
import pycocotools.mask as maskUtils
from ..registry import PIPELINES


'''
pip install SimpleITK

sitk image: (width, height, depth)
sitk ndarray: (depth, height, width)
'''


@PIPELINES.register_module
class LoadDicomSingle(object):

    def __init__(self):
        pass

    def __call__(self, results):
        if results['data_root'] is not None:
            dicom = osp.join(results['data_root'], results['img_info']['dicom'])
        else:
            dicom = results['img_info']['dicom']

        itk_img = sitk.ReadImage(dicom, sitk.sitkFloat32)
        input_data = sitk.GetArrayFromImage(itk_img)[0]  # (height, width)

        results['dicom'] = (dicom,)
        results['input'] = input_data
        results['ori_shape'] = input_data.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class LoadDicomDouble(object):

    def __init__(self):
        pass

    def __call__(self, results):
        if results['data_root'] is not None:
            dicom_A = osp.join(results['data_root'], results['img_info']['dicom_A'])
            dicom_B = osp.join(results['data_root'], results['img_info']['dicom_B'])
        else:
            dicom_A = results['img_info']['dicom_A']
            dicom_B = results['img_info']['dicom_B']

        itk_img = sitk.ReadImage(dicom_A, sitk.sitkFloat32)
        input_data = sitk.GetArrayFromImage(itk_img)[0]  # (height, width)

        itk_img = sitk.ReadImage(dicom_B, sitk.sitkFloat32)
        target_data = sitk.GetArrayFromImage(itk_img)[0]  # (height, width)

        results['dicom'] = (dicom_A, dicom_B)
        results['input'] = input_data
        results['target'] = target_data
        results['ori_shape'] = input_data.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self, with_bbox=False, with_mask=False, poly2mask=True):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.poly2mask = poly2mask

    def _load_boxes(self, results):
        results['gt_boxes'] = results['ann_info']['boxes']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['ori_shape']
        gt_masks = results['ann_info']['masks']

        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks

        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_boxes(results)

        if self.with_mask:
            results = self._load_masks(results)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(with_bbox={}, with_mask={})'.format(self.with_bbox, self.with_mask)
