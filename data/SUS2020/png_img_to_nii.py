#!/usr/bin/env python
# coding: utf-8

# ## Converting images from png to nii file
# 
# 
# ### Overview
# 
# This is step 0 for data preparation
# 
# Input: images in `.png` format
# 
# Output: images in `.nii` format, indexed by patient id

# In[1]:


import os
import glob

import numpy as np
import PIL
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
sys.path.insert(0, '../../dataloaders/')
import niftiio as nio
import nibabel as nib

isDWI = False
example = "/home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/FINAL_Najm_v21-0.5/CTP_01_001/01/01.tiff" # example of ground-truth file

# search for scan ids

# root_path = 'D:/Preprocessed-SUS2020_v2/TEST_v21-0.25/'
root_path = "/home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/FINAL_Najm_v21-0.5/"
# root_path = "/home/prosjekt/PerfusionCT/StrokeSUS/DWI/REGISTERED_2.0/Studies/"

ids = os.listdir(root_path)
OUT_DIR = "/home/stud/lucat/PhD_Project/ADNet/nii_studies/"
# OUT_DIR = '/bhome/lucat/nii_studies/'
# OUT_DIR = '/bhome/lucat/DWI_nii_studies/'
# OUT_DIR = "D:/niis/SUS2020/"


print(ids)


# Write them to nii files for the ease of loading in future
for curr_id in ids:
    timefold = glob.glob(f'{root_path}{curr_id}/*')
    if isDWI:
        pngs = sorted(timefold, key=lambda x: int(os.path.basename(x).split(".png")[0]))
        buffer = []
        for fid in pngs: buffer.append(PIL.Image.open(fid))
        vol = np.stack(buffer, axis=0)
        sitk.WriteImage(sitk.GetImageFromArray(vol), f'{OUT_DIR}/study_{curr_id}.nii.gz')
    else:
        timefold = sorted(timefold, key=lambda x: int(os.path.basename(x)))

        for t, curr_f in enumerate(timefold):
            buffer = []
            tiffs = glob.glob(f'{curr_f}/*.tiff')
            tiffs = sorted(tiffs, key=lambda x: int(os.path.basename(x).split(".tiff")[0]))
            for fid in tiffs: buffer.append(PIL.Image.open(fid))
            vol = np.stack(buffer, axis=0)
            if not os.path.isdir(f'{OUT_DIR}/study_{curr_id}/'): os.makedirs(f'{OUT_DIR}/study_{curr_id}/')
            sitk.WriteImage(sitk.GetImageFromArray(vol), f'{OUT_DIR}/study_{curr_id}/{str(t)}.nii.gz')
    print(f'Study with id {curr_id} has been saved!')





