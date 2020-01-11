# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Gregoire Dauce (https://git.ee.ethz.ch/dauceg.student.ethz.ch)
# Robin Brügger

import os
import numpy as np
import logging
import string
import gc
import h5py
from skimage import transform
import sys

sys.path.append('/home/dell/github/bright-light-515_state_of_the_art_models')
import utils
import cv2
import warnings
import SimpleITK as sitk  # If you can't import this then run "conda install -c simpleitk simpleitk"
from nipype.interfaces.ants import N4BiasFieldCorrection
from skimage.exposure import equalize_adapthist, equalize_hist

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5
alpha_dic = {ch: n for n, ch in enumerate(string.ascii_uppercase)}


def test_train_val_split(patient_id):
    if patient_id % 10 >= 4:
        return 'train'
    elif patient_id % 10 >= 2:
        return 'validation'
    else:
        return 'train'
    # return 'train'


def normalise_image(image):
    '''
    standardize based on nonzero pixels
    '''
    m = np.nanmean(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    s = np.nanstd(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    normalized = np.divide((image - m), s)
    image = np.where(image == 0, 0, normalized)
    return image


def crop_volume_allDim(image, mask=None):
    '''
    Strip away the zeros on the edges of the three dimensions of the image
    Idea: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    '''
    coords = np.argwhere(image > 0)
    x0, y0, z0, _ = coords.min(axis=0)
    x1, y1, z1, _ = coords.max(axis=0) + 1

    image = image[x0:x1, y0:y1, z0:z1, :]
    if not mask is None:
        return image, mask[x0:x1, y0:y1, z0:z1]
    return image


def crop_or_pad_slice_to_size(image, target_size, channels=None, offset=None):
    '''
    Make sure that the image has the desired dimensions
    '''

    if offset is None:
        offset = (0, 0, 0)

    x_t, y_t, z_t = target_size[0:3]
    x_s, y_s, z_s = image.shape[0:3]

    if not channels is None:
        output_volume = np.zeros((x_t, y_t, z_t, channels), dtype=np.float32)
    else:
        output_volume = np.zeros((x_t, y_t, z_t), dtype=np.float32)

    x_d = abs(x_t - x_s) // 2 + offset[0]
    y_d = abs(y_t - y_s) // 2 + offset[1]
    z_d = abs(z_t - z_s) // 2 + offset[2]

    t_ranges = []
    s_ranges = []

    for t, s, d in zip([x_t, y_t, z_t], [x_s, y_s, z_s], [x_d, y_d, z_d]):

        if t < s:
            t_range = slice(t)
            s_range = slice(d, d + t)
        else:
            t_range = slice(d, d + s)
            s_range = slice(s)

        t_ranges.append(t_range)
        s_ranges.append(s_range)

    if not channels is None:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2], :] = image[s_ranges[0], s_ranges[1], s_ranges[2], :]
    else:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2]] = image[s_ranges[0], s_ranges[1], s_ranges[2]]

    return output_volume


def img_resize(imgs, size, equalize=False):
    img_rows = size[1]
    img_cols = size[2]
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def prepare_data(input_folder, output_file, size, input_channels, target_resolution):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    if len(size) != 3:
        raise AssertionError('Inadequate number of size parameters')
    if len(target_resolution) != 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    hdf5_file = h5py.File(output_file, "w")

    file_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Counting files and parsing meta data...')

    for i in range(50):
        print(input_folder + 'Case' + "%02d" % i)
        train_test = test_train_val_split(i)
        file_list[train_test].append(input_folder + 'Case' + "%02d" % i)
    for i in range(30):
        print('/home/dell/data/Dataset/prostate_raw/TestData/' + 'Case' + '%02d' % i)
        file_list['test'].append('/home/dell/data/Dataset/prostate_raw/TestData/' + 'Case' + '%02d' % i)

    n_train = len(file_list['train'])
    n_test = len(file_list['test'])
    n_val = len(file_list['validation'])

    print('Debug: Check if sets add up to correct value:')
    print(n_train, n_val, n_test, n_train + n_val + n_test)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            print([num_points] + list(size) + [input_channels])
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt,
                                                              [num_points] + list(size) + [input_channels],
                                                              dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Parsing image files')

    for train_test in ['train', 'test', 'validation']:

        write_buffer = 0
        counter_from = 0

        for folder in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % folder)

            patient_id = folder
            if not train_test == 'test':
                mask = sitk.ReadImage(folder + '_segmentation.mhd')
                mask = sitk.GetArrayFromImage(mask)
                mask = img_resize(mask, size=size)
                mask = crop_or_pad_slice_to_size(mask, target_size=size)
                mask_list[train_test].append(mask)
            img = sitk.ReadImage(folder + '.mhd')
            img = sitk.GetArrayFromImage(img)
            print(f'The shape of {patient_id} is {img.shape}!!!')
            img = normalise_image(img)
            img = img_resize(img, size=size)
            img = crop_or_pad_slice_to_size(img, target_size=size)
            print(f'After resize shape is {img.shape}!!!')
            img = img[:, :, :, np.newaxis]

            img_list[train_test].append(img)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:
                counter_to = counter_from + write_buffer
                if train_test == 'test':
                    _write_range_to_hdf5(data, train_test, img_list, counter_from=counter_from, counter_to=counter_to)
                    _release_tmp_memory(img_list, None, train_test)
                else:
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        if len(file_list[train_test]) > 0:
            _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)

    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list=None, counter_from=0, counter_to=0):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    # img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    # mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_list[train_test]
    if mask_list is not None:
        hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_list[train_test]


def _release_tmp_memory(img_list, mask_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    if mask_list is not None:
        mask_list[train_test].clear()
    gc.collect()


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                input_channels,
                                target_resolution,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'N4bias_data_3D_size_%s_res_%s_fold5.hdf5' % (size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, size, input_channels, target_resolution)
        # bias_normalization(input_folder, data_file_path)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    input_folder = "/home/dell/data/Dataset/prostate_raw/TrainingData_Part1/"
    preprocessing_folder = "/home/dell/data/Dataset/prostate_raw/"
    # bias_normalization(input_folder, preprocessing_folder)

    d = load_and_maybe_process_data(input_folder, preprocessing_folder, (48, 256, 256), 1, (1.0, 1.0, 1.0),
                                    force_overwrite=True)

