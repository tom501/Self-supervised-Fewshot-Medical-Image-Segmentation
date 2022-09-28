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


example = "/home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/Parametric_Maps/CTP_01_001/*/*/01.png"  # example of ground-truth file

# search for scan ids

# root_path = 'D:/Preprocessed-SUS2020_v2/TEST_v21-0.25/'
root_path = "/home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/Parametric_Maps/"

ids = os.listdir(root_path)
OUT_DIR = './PMs_niis/'
colorbar_coord = (129, 435)

print(ids)

pmlist = ["CBF", "CBV", "TTP", "TMAX", "MIP"]

# Write them to nii files for the ease of loading in future
for curr_id in ids:
    timefolds = glob.glob(f'{root_path}{curr_id}/*')
    for tf in timefolds:
        timefold = glob.glob(f'{tf}/*')
        if len(timefold)<len(pmlist): continue

        for t, curr_f in enumerate(timefold):
            if os.path.basename(curr_f) not in pmlist: continue
            buffer = []
            pngs = glob.glob(f'{curr_f}/*.png')
            pngs = sorted(pngs, key=lambda x: int(os.path.basename(x).split(".png")[0]))
            for fid in pngs:
                img = PIL.Image.open(fid)
                img = np.asarray(img)
                img[:,colorbar_coord[1]:] = 0
                buffer.append(img)
            vol = np.stack(buffer, axis=0)
            if not os.path.isdir(f'{OUT_DIR}/{curr_id}/'): os.makedirs(f'{OUT_DIR}/{curr_id}/')
            sitk.WriteImage(sitk.GetImageFromArray(vol), f'{OUT_DIR}/{curr_id}/{os.path.basename(curr_f)}.nii.gz')
        print(f'Study with id {curr_id} has been saved!')




# In[ ]:




