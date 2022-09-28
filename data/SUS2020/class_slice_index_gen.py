#!/usr/bin/env python
# coding: utf-8

# ## Generate class-slice indexing table for experiments
# 
# 
# ### Overview
# 
# This is for experiment setting up for simulating few-shot image segmentation scenarios
# 
# Input: pre-processed images and their ground-truth labels
# 
# Output: a `json` file for class-slice indexing

# In[2]:


import numpy as np
import os
import glob
import SimpleITK as sitk
import sys
import json

sys.path.insert(0, '../../dataloaders/')
import niftiio as nio

# In[4]:


SEG_BNAME = "./niis/label_*.nii.gz"

# In[5]:


segs = glob.glob(SEG_BNAME)
segs = [fid for fid in sorted(segs, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0]))]

# In[7]:


segs

# In[13]:


classmap = {}
LABEL_NAME = ["background", "core"]

MIN_TP = 100  # minimum number of positive label pixels to be recorded. Use >100 when training with manual annotations for more stable training

fid = f'./niis/classmap_{MIN_TP}.json'  # name of the output file.
for _lb in LABEL_NAME:
    classmap[_lb] = {}
    for _sid in segs:
        pid = _sid.split("label_")[-1].split(".nii.gz")[0]
        classmap[_lb][pid] = []

for seg in segs:
    pid = seg.split("label_")[-1].split(".nii.gz")[0]
    lb_vol = nio.read_nii_bysitk(seg)
    n_slice = lb_vol.shape[0]
    for slc in range(n_slice):
        for cls in range(len(LABEL_NAME)):
            if cls in lb_vol[slc, ...]:
                if np.sum(lb_vol[slc, ...]) >= MIN_TP:
                    classmap[LABEL_NAME[cls]][str(pid)].append(slc)
    print(f'pid {str(pid)} finished!')

with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
    fopen.close()
