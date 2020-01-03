import os
import sys
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
from collections import defaultdict


'''
https://www.dicomstandard.org/current/
- DICOM Part 6: Data Dictionary

0008|0060 - Modality(MRI/CT/CR/DR)
0008|0070 - Manufacturer
0018|0015 - Body Part Examined
0008|1030 - Study Description
0010|0020 - Patient ID
'''


G_TAGS = ['0008|0060', '0008|0070', '0018|0015', '0008|1030']


def get_series_ids(data_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(data_dir)
    return series_ids


def get_series_files(data_dir, series_id):
    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(data_dir, series_id)
    return series_files


def meta_data_from_file(dicom_path, tags=None):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_path)
    reader.ReadImageInformation()

    if tags is None:
        tags = G_TAGS

    vals = [reader.GetMetaData(tag) for tag in tags if reader.HasMetaDataKey(tag)]
    return [v.split()[0].upper() for v in vals]


def copy_files(file_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in file_list:
        if file_path[-4:] != '.dcm':
            filename = os.path.basename(file_path) + '.dcm'
            shutil.copy(file_path, os.path.join(output_dir, filename))
        else:
            shutil.copy(file_path, output_dir)


def split_dir(input_dir, output_dir):
    cache = Path(input_dir).glob('**/*')
    cache = set([i.parent.as_posix() for i in cache])

    shutil.rmtree(output_dir, ignore_errors=True)

    counts = defaultdict(dict)
    patients = defaultdict(set)
    for data_dir in sorted(cache):
        for series_id in get_series_ids(data_dir):
            dicom_list = get_series_files(data_dir, series_id)

            if len(dicom_list) < 3:
                continue

            sub_dir = '_'.join(meta_data_from_file(dicom_list[0])[:3])
            patient_id = meta_data_from_file(dicom_list[0], ['0010|0020'])[0]
            copy_files(dicom_list, os.path.join(output_dir, sub_dir, series_id))

            counts[sub_dir]['images'] = counts[sub_dir].get('images', 0) + len(dicom_list)
            counts[sub_dir]['series'] = counts[sub_dir].get('series', 0) + 1
            patients[sub_dir].add(patient_id)

    for sub_dir, patient_ids in patients.items():
        counts[sub_dir]['patients'] = len(patient_ids)

    msg = ['# ReadMe', '', '## counts', json.dumps(counts, indent=4, sort_keys=True)]
    with open(os.path.join(output_dir, 'readme.md'), 'w') as f:
        f.write('\n'.join(msg))
    print('\n'.join(msg))


if __name__ == '__main__':
    split_dir(sys.argv[1], sys.argv[2])
