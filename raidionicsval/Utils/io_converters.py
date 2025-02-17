import logging
import os
import pickle
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Tuple, List
from PIL import Image
import SimpleITK as sitk


def get_fold_from_file(filename, fold_number):
    """
    Return the content of the validation and test folds for the current fold number by parsing the specified file.
    :param filename: lists of samples used for training and specifying the folds' content.
    :param fold_number: fold from which the sets' content should be collected.
    :return: two lists of strings containing the patients' names featured in the validation and test folds.
    """
    val_set = None
    test_set = None
    if filename is None or not os.path.exists(filename):
        raise AttributeError(f'The provided filename containing the folds distribution is invalid: {filename}.\n')

    if filename.split('.')[-1] == 'txt':
        with open(filename) as f:
            for i in range(fold_number):
                val_line = f.readline()
                test_line = f.readline()

            val_line = f.readline().strip()
            test_line = f.readline().strip()

            val_set = val_line.split(' ')
            val_set = [x for x in val_set]
            test_set = test_line.split(' ')
            test_set = [x for x in test_set]
    elif filename.split('.')[-1] == 'pkl':
        folds_file = open(filename, 'rb')
        folds = pickle.load(folds_file)

        val_set = folds[int(fold_number)]['val']
        test_set = folds[int(fold_number)]['val']
        if 'test' in folds[int(fold_number)].keys():
            test_set = folds[int(fold_number)]['test']

    return val_set, test_set


def reload_optimal_validation_parameters(study_filename):
    """
    Load the optimal probability and Dice thresholds identified during the validation process, from the
    optimal_dice_study.csv file located inside the validation folder.
    :param study_filename: filename to the optimal_dice_study.csv file located inside the validation folder.
    :return: two floats, the best probability threshold and best Dice threshold.
    """
    study_df = pd.read_csv(study_filename)
    optimums = study_df.iloc[-1]

    return optimums['Detection threshold'], optimums['Dice threshold']

def open_image_file(input_filename: str) -> Tuple[np.ndarray, str, List]:
    ext = '.' + '.'.join(input_filename.split('.')[1:])
    input_array = None
    input_specifics = []

    if ext == ".nii" or ext == ".nii.gz":
        input_ni = nib.load(input_filename)
        if len(input_ni.shape) == 4:
            input_ni = nib.four_to_three(input_ni)[0]
        input_array = input_ni.get_fdata()[:]
        input_specifics = [input_ni.affine, input_ni.header.get_zooms()]
    elif ext in [".tif", ".tiff", ".png"]:
        input_array = Image.open(input_filename)
        input_specifics = [np.eye(4, dtype=int), [1., 1.]]
    elif ext == ".mhd":
        # To fix problem with TransformMatrix = 1 0 0 0 1 0 0 0 1 for 2D images
        # Reading mhd file and creating a temporary copy without TransformMatrix = 1 0 0 0 1 0 0 0 1
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        filtred_lines = [line for line in lines if not line.startswith('TransformMatrix')]
        temp_file_path = input_filename.replace('.mhd', '_temp.mhd')
        with open(temp_file_path, 'w') as ft:
            ft.writelines(filtred_lines)
        input_img = sitk.ReadImage(temp_file_path)
        input_array = sitk.GetArrayFromImage(input_img)
        input_specifics = [input_img.GetDirection(), input_img.GetSpacing()]
        os.remove(temp_file_path)
    else:
        logging.error("Working with an unknown file type: {}. Skipping...".format(ext))

    return input_array, ext, input_specifics


def save_image_file(output_array, output_filename: str, specifics: List = None) -> None:
    ext = '.'.join(output_filename.split('.')[1:])

    if ext == ".nii" or ext == ".nii.gz":
        nib.save(nib.Nifti1Image(output_array, affine=specifics[0]), output_filename)
    elif ext in [".tif", ".tiff", ".png"]:
        Image.fromarray(output_array).save(output_filename)
    elif ext == ".mhd":
        image = sitk.GetImageFromArray(output_array)
        image.SetSpacing(specifics[1])
        image.SetDirection(specifics[0])
        image.SetOrigin([0.0,0.0])
        sitk.WriteImage(image, output_filename)
    else:
        logging.error("Working with an unknown file type: {}. Skipping...".format(ext))
