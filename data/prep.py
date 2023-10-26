import os

import h5py
import numpy as np
from pathlib import Path

from utils import read_frame_transform
from config import ROOT
import pandas as pd



FLAG_SCANFILE = False

DIR_RAW = ROOT / "forearm_US"
NUM_SCANS = 12
RESAMPLE_FACTOR = 4
PATH_SAVE = ROOT
DELAY_TFORM = 4  # delayed tform from temporal calibration
fh5_path = PATH_SAVE / f'frames_res{RESAMPLE_FACTOR}.h5'
if fh5_path.is_file():
    os.remove(fh5_path)

fh5_frames = h5py.File(fh5_path,'a') 
if FLAG_SCANFILE:
    fh5_scans = h5py.File( PATH_SAVE/ f'scans_res{RESAMPLE_FACTOR}.h5','a')

folders_subject = [f for f in DIR_RAW.iterdir() if f.is_dir()]
num_frames = np.zeros((len(folders_subject), NUM_SCANS), dtype=np.uint16)
for i_sub, folder in enumerate(folders_subject):

    fn_csv = list(folder.glob('*.csv'))
    if len(fn_csv) != 1: 
        raise Exception('Should contain 1 csv file in folder "{}"'.format(folder))

    scan_crop_idx = pd.read_csv(folder / fn_csv[0], index_col=0)
    # scan_crop_idx = read_scan_crop_indices_file(os.path.join(DIR_RAW, folder, fn_xls[0]), NUM_SCANS)  # TBA: checks for 1) item name/order and if the 12 scans are complete

    fn_mha = list(folder.glob('*.mha'))
    fn_mha.sort(reverse=False)  # ordered in acquisition time
    if len(fn_mha) != NUM_SCANS: raise('Should contain 12 mha files in folder "{}"'.format(folder))    

    for i_scan, fn in enumerate(fn_mha):

        frames, tforms, _ = read_frame_transform(
            filename = fn,
            scan_crop_indices = list(scan_crop_idx.loc[fn.name]),
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )
        tforms_inv = np.linalg.inv(tforms)  # pre-compute the iverse

        num_frames[i_sub, i_scan] = frames.shape[0]
        for i_frame in range(frames.shape[0]):
            fh5_frames.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames.shape[1:3], dtype=frames.dtype, data=frames[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms_inv[i_frame,...])
        if FLAG_SCANFILE:
            fh5_scans.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames.shape, dtype=frames.dtype, data=frames)
            fh5_scans.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms)
            fh5_scans.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms_inv)

fh5_frames.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
fh5_frames.create_dataset('sub_folders', len(folders_subject), data=[f.name for f in folders_subject])
fh5_frames.create_dataset('frame_size', 2, data=frames.shape[1:3])
fh5_frames.flush()
fh5_frames.close()

if FLAG_SCANFILE:
    fh5_scans.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
    fh5_scans.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans.create_dataset('frame_size', 2, data=frames.shape[1:3])
    fh5_scans.flush()
    fh5_scans.close()

print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"))
