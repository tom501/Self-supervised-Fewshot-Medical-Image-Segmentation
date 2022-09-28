#!/usr/bin/env python
# coding: utf-8

# ## Generate superpixel-based pseudolabels
#
#
# ### Overview
#
# This is the third step for data preparation
#
# Input: normalized images
#
# Output: pseulabel label candidates for all the images

# In[1]:


import matplotlib.pyplot as plt
import copy
import skimage

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import label
import scipy.ndimage.morphology as snm
from sklearn.metrics import jaccard_score
from scipy.ndimage import variance
from skimage import io
import argparse
import numpy as np
import glob, time, os, PIL, multiprocessing
from multiprocessing import Array
import pickle as pkl
from felzenszwalb_3d import felzenszwalb_3d

import SimpleITK as sitk

to01 = lambda x: (x - x.min()) / (x.max() - x.min())

# **Summary**
#
# a. Generate a mask of the patient to avoid pseudolabels of empty regions in the background
#
# b. Generate superpixels as pseudolabels
#
# **Configurations of pseudlabels**
#
# ```python
# # default setting of minimum superpixel sizes
# segs = seg_func(img[ii, ...], min_size = 400, sigma = 1)
# # you can also try other configs
# segs = seg_func(img[ii, ...], min_size = 100, sigma = 0.8)
# ```
#

# In[2]:

HOME = "/home/student/lucat/PhD_Project/Self-supervised-Fewshot-Medical-Image-Segmentation/data"
DATASET_CONFIG = {
    'SABS': {
        'img_bname': f'../SABS/sabs_CT_normalized/image_*.nii.gz',
        'out_dir': '../SABS/sabs_CT_normalized',
        'fg_thresh': 1e-4,
        "method": "felzenszwalb"
    },
    'CHAOST2': {
        'img_bname': f'../CHAOST2/chaos_MR_T2_normalized/image_*.nii.gz',
        'out_dir': '../CHAOST2/chaos_MR_T2_normalized',
        'fg_thresh': 1e-4 + 50,
        "method": "felzenszwalb"
    },
    'SUS2020': {
        'img_bname': f'../SUS2020/niis/study_*/',
        'out_dir': '../SUS2020/SLIC_supervoxel/',
        'fg_thresh': 1e-4+15,
        "method": "slic",
        'out_png_dir': '../SUS2020/SLIC_pngs/',
        "n_segments": 100,
        "compactness": 1
    },
    'SUS2020_PMs': {
        'img_bname': f'../SUS2020/PMs_niis/*/',
        'out_dir': '../SUS2020/SLIC_PMs/placeholder/',
        'fg_thresh': 1e-4,
        "method": "slic",
        'out_png_dir': '../SUS2020/SLIC_PMs_pngs/placeholder/',
        "n_segments": 100,
        "compactness": 10,
        # "mask_fg": "/home/prosjekt/PerfusionCT/StrokeSUS/DWI/REGISTERED/COMBINED/GT/"
    },
    'SUS2020_PMs_RGB': {
        'img_bname': f'{HOME}/SUS2020/PMs_niis/*/',
        'out_dir': f'{HOME}/SUS2020/FELZENSZWALB_PMs_RGB/placeholder/',
        'out_png_dir': f'{HOME}/SUS2020/FELZENSZWALB_PMs_RGB_pngs/placeholder/',
        'fg_thresh': 1e-4+1,  # 1e-4+100 <-- for penumbra+core mask # 1e-4+170 <-- for penumbra # 1e-4+220 <-- for core
        # "penumbra": True,
        # "core": True,
        "method": "felzenszwalb_3d",  # "slic",
        "n_segments": 100,
        "compactness": [10],  # [0.01,0.10,1,10,100],
        "min_size": [1500,2000,3000],
        # "mask_fg": "/home/prosjekt/PerfusionCT/StrokeSUS/DWI/REGISTERED/COMBINED/GT/",
        "mask_fg": "/home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/GT_TIFF/",
        "mask_ctp": True,
    }
}

slic_zero = True

DOMAIN = 'SUS2020_PMs_RGB'
img_bname = DATASET_CONFIG[DOMAIN]['img_bname']
studies = glob.glob(img_bname)
out_dir = DATASET_CONFIG[DOMAIN]['out_dir']
out_png_dir = DATASET_CONFIG[DOMAIN]['out_png_dir']
n_segments = DATASET_CONFIG[DOMAIN]['n_segments']
compactness = DATASET_CONFIG[DOMAIN]['compactness']
min_size = DATASET_CONFIG[DOMAIN]['min_size']

if "SUS2020_PMs" not in DOMAIN: studies = np.unique([x.split('study_')[-1][:10] for x in studies])
else: studies = np.unique([x.split("PMs_niis/")[-1][:10] for x in studies])
pmlist = ["TTP", "TMAX", "MIP", "CBF", "CBV"]


# wrapper for process 3d image in 2d
def superpix_vol(img, what, method='felzenszwalb', **kwargs):
    """
    loop through the entire volume
    assuming image with axis z, x, y
    """
    out_vol = np.zeros(img.shape) if method!="felzenszwalb" else np.zeros(img[...,-1].shape)
    channel_axis = None if "SUS2020_PMs" not in DOMAIN else -1

    if method=="felzenszwalb":
        seg_func = skimage.segmentation.felzenszwalb
        for ii in range(img.shape[0]): out_vol[ii, ...] = seg_func(img[ii, ...], min_size=400, sigma=0, channel_axis=channel_axis)
    elif method=="felzenszwalb_3d": out_vol = felzenszwalb_3d(img, scale=1, sigma=0, min_size=int(what), spacing=(5.,0.44,0.44))
    elif method == "slic":
        convert2lab = False if "SUS2020_PMs" not in DOMAIN else True

        # out_vol = slic(img, n_segments=n_segments, compactness=what, max_iter=10, sigma=0, enforce_connectivity=True,
        #                multichannel=channel_axis, convert2lab=convert2lab, spacing=(5.,0.44,0.44), slic_zero=slic_zero)  # convert2lab=True --> channel_axis is not None  && image.shape[-1] == 3
        out_vol = slic(img, n_segments=n_segments, compactness=what, max_num_iter=20, sigma=0, enforce_connectivity=True,
                       channel_axis=channel_axis, convert2lab=convert2lab, spacing=(5.,0.44,0.44), slic_zero=slic_zero)  # convert2lab=True --> channel_axis is not None  && image.shape[-1] == 3
    else: raise NotImplementedError

    return out_vol


# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh):  # change this by your need
    if "mask_ctp" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["mask_ctp"] and img_2d.max()>300: img_2d = np.divide(img_2d, 255.)
    if "penumbra" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["penumbra"]:
        t_1 = np.float32(img_2d>thresh-20)
        t_2 = np.float32(img_2d<thresh+20)
        mask_map = np.float32((t_1+t_2)==2)
    else: mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999: return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return np.array(fill_mask,dtype=np.float32)


# remove superpixels within the empty regions
def superpix_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = np.max(lbvs)
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append(max_lb)
    orig_raw_seg2d = raw_seg2d
    raw_seg2d = raw_seg2d*mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    tmp_seg = np.zeros(orig_raw_seg2d.shape)
    overlap_seg2d = np.zeros(orig_raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0: continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            if lb_new in np.unique(out_seg2d): tmp_seg[orig_raw_seg2d==lbv] = lb_new
            lb_new += 1

    tmp = label(tmp_seg)
    for lbv in lbvs:
        if lbv == 0: continue
        tm = (raw_seg2d == lbv)
        uniq_vals = np.unique(tmp * tm)
        for cc in uniq_vals:
            if cc != 0: overlap_seg2d[tmp == cc] = lbv
    return out_seg2d, overlap_seg2d


def superpix_wrapper(img, what, verbose=False, fg_thresh=1e-4, method="felzenszwalb", stud=""):
    if method=="slic" and "SUS2020_PMs" not in DOMAIN:  # reshape the volume
        new_img = np.zeros((img.shape[1],img.shape[2],img.shape[0]))
        for ii in range(img.shape[0]): new_img[...,ii] = img[ii,...]
        img = new_img
    elif method=="felzenszwalb_3d":
        new_img = np.zeros(img[...,-1].shape)
        for x in range(img.shape[0]): new_img[x,...] = skimage.color.rgb2gray(img[x,...])
        img = new_img
    raw_seg = superpix_vol(img, what, method=method)  # get the superpixel
    # Initialize foreground mask and segment volume
    fg_mask_vol = np.zeros(raw_seg.shape)
    processed_seg_vol = np.zeros(raw_seg.shape)
    seg_vol_overlap = np.zeros(raw_seg.shape)
    processed_seg_vol_rgb = np.zeros(raw_seg.shape+(3,))

    if "felzenszwalb" in method:
        for ii in range(raw_seg.shape[0]):
            # if verbose: print("doing {} slice".format(ii))
            if "mask_fg" in DATASET_CONFIG[DOMAIN].keys() or "RGB" in DOMAIN:
                idx = str(ii + 1)
                if len(idx) == 1: idx = "0" + idx
                suffix = "tiff" if DATASET_CONFIG[DOMAIN]["mask_ctp"] else "png"
                mask_path = os.path.join(DATASET_CONFIG[DOMAIN]["mask_fg"], f'{stud}/{idx}.{suffix}')
                if not os.path.isfile(mask_path): return None, None
                mask = np.asarray(PIL.Image.open(mask_path))
                if len(mask.shape) > 2: mask = mask[:, :, 1]
            else: mask = img[ii, ...]
            fg_mask_vol[ii] = fg_mask2d(mask, fg_thresh)  # get a binary mask of the patient with fg_thresh
            processed_seg_vol[ii], seg_vol_overlap[ii] = superpix_masking(raw_seg[ii, ...], fg_mask_vol[ii])  # remove superpixels within the empty regions
            if "RGB" in DOMAIN:
                processed_seg_vol_rgb[ii] = skimage.color.label2rgb(processed_seg_vol[ii], img[ii,...], kind='avg')
                processed_seg_vol_rgb[ii] *= np.stack([fg_mask_vol[ii],fg_mask_vol[ii],fg_mask_vol[ii]],axis=-1)
    elif method=="slic":
        if "mask_fg" in DATASET_CONFIG[DOMAIN].keys() or "RGB" in DOMAIN:
            for ii in range(raw_seg.shape[0]):
                if "mask_fg" in DATASET_CONFIG[DOMAIN].keys():
                    idx = str(ii+1)
                    if len(idx) == 1: idx = "0" + idx
                    suffix = "tiff" if DATASET_CONFIG[DOMAIN]["mask_ctp"] else "png"
                    mask_path = os.path.join(DATASET_CONFIG[DOMAIN]["mask_fg"],f'{stud}/{idx}.{suffix}')
                    if not os.path.isfile(mask_path): return None, None
                    mask = np.asarray(PIL.Image.open(mask_path))
                    if len(mask.shape)>2: mask = mask[:,:,1]
                else: mask = skimage.color.rgb2gray(img[ii,...])
                fg_mask_vol[ii] = fg_mask2d(mask, fg_thresh)  # get a binary mask of the patient with fg_thresh
                processed_seg_vol[ii], seg_vol_overlap[ii] = superpix_masking(raw_seg[ii], fg_mask_vol[ii])  # remove superpixels within the empty regions
                if "RGB" in DOMAIN:
                    processed_seg_vol_rgb[ii] = skimage.color.label2rgb(processed_seg_vol[ii], img[ii,...], kind='avg')
                    processed_seg_vol_rgb[ii] *= np.stack([fg_mask_vol[ii],fg_mask_vol[ii],fg_mask_vol[ii]],axis=-1)
        else:
            mask = img[...,0]
            fg_mask_vol = fg_mask2d(mask, fg_thresh)  # get a binary mask of the patient with fg_thresh
            processed_seg_vol, seg_vol_overlap = superpix_masking(raw_seg, fg_mask_vol)  # remove superpixels within the empty regions

    if "RGB" in DOMAIN: processed_seg_vol = processed_seg_vol_rgb
    return fg_mask_vol, processed_seg_vol, seg_vol_overlap


# copy spacing and orientation info between sitk objects
def copy_info(src, dst, same_dim=True):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    if same_dim: dst.SetDirection(src.GetDirection())
    # dst.CopyInfomation(src)
    return dst


def strip_(img, lb):
    img = np.int32(img)
    if isinstance(lb, float):
        lb = int(lb)
        return np.float32(img == lb) * float(lb)
    elif isinstance(lb, list):
        out = np.zeros(img.shape)
        for _lb in lb: out += np.float32(img == int(_lb)) * float(_lb)
        return out
    else: raise Exception


def single_process(stud, what):
    timefold = []
    if DOMAIN == "SUS2020":
        timefold = glob.glob(f'{HOME}/SUS2020/niis/study_{stud}/*')
        timefold = sorted(timefold, key=lambda x: int(os.path.basename(x.split(".nii.gz")[0])))
    elif "SUS2020_PMs" in DOMAIN:
        timefold = glob.glob(f'{HOME}/SUS2020/PMs_niis/{stud}/*')

    print(f'patient {stud} has started in {out_png_dir}')
    for t, img_fid in enumerate(timefold):
        if "SUS2020_PMs" in DOMAIN:
            idx = os.path.basename(img_fid).split(".nii.gz")[0]
            if idx not in pm: continue
        else:
            idx = os.path.basename(img_fid).split(".nii.gz")[0]
            superpix = f"superpix-{MODE}_{idx}"
            fgmask = f"fgmask_{idx}"
        im_obj = sitk.ReadImage(img_fid)

        out_fg, out_seg, overlap_seg = superpix_wrapper(sitk.GetArrayFromImage(im_obj), what=what, verbose=True, stud=stud,
                                                        fg_thresh=DATASET_CONFIG[DOMAIN]['fg_thresh'],
                                                        method=DATASET_CONFIG[DOMAIN]["method"])
        if out_fg is None and out_seg is None: continue
        loop = 1 if len(out_fg.shape) == 2 else out_fg.shape[0]
        for i in range(loop):
            if loop > 1:
                idx = str(i + 1)
                if len(idx) == 1: idx = "0" + idx
                superpix = f"superpix-{MODE}_{idx}"
                overlappix = f"overlap-{MODE}_{idx}"
                fgmask = f"fgmask_{idx}"
            # Save the png images
            if not os.path.isdir(f'{out_png_dir}/{stud}/'): os.makedirs(f'{out_png_dir}/{stud}/')
            seg_fid = os.path.join(out_png_dir, f'{stud}/{superpix}.png')
            overlap_fid = os.path.join(out_png_dir, f'{stud}/{overlappix}.png')
            msk_fid = os.path.join(out_png_dir, f'{stud}/{fgmask}.png')
            # extract img for convert to RGB
            out_fg_tmp = out_fg[i] if loop > 1 else out_fg
            out_seg_tmp = out_seg[i] if loop > 1 else out_seg
            overlap_seg_tmp = overlap_seg[i] if loop > 1 else overlap_seg
            union_mask = np.int32(overlap_seg_tmp > 0)
            out_fg_i = PIL.Image.fromarray((out_fg_tmp * 255).astype(np.uint8))
            if out_fg_i.mode == "F": out_fg_i = out_fg_i.convert("L")
            mult = 255 if "RGB" not in DOMAIN or DATASET_CONFIG[DOMAIN]["method"] == "felzenszwalb_3d" else 1
            if "mask_fg" in DATASET_CONFIG[DOMAIN].keys():
                # v = variance(out_seg_tmp*mult)/(np.sum(out_fg_tmp)+1e-4)
                v = jaccard_score(out_fg_tmp, union_mask, average="micro", zero_division=0.0)
                indexelem = dictelems[stud][i]
                iou_dict_shared[pm][indexelem] = v
                labels_dict_shared[pm][indexelem] = np.sum(np.unique(overlap_seg_tmp) != 0)
            out_seg_i = PIL.Image.fromarray((out_seg_tmp * mult).astype(np.uint8))
            overlap_seg_i = PIL.Image.fromarray((overlap_seg_tmp * 255).astype(np.uint8))
            if out_seg_i.mode == "F": out_seg_i = out_seg_i.convert("L")
            if overlap_seg_i.mode == "F": overlap_seg_i = overlap_seg_i.convert("L")
            # save
            out_fg_i.save(msk_fid)
            out_seg_i.save(seg_fid)
            overlap_seg_i.save(overlap_fid)
            # Save the nii file
            out_fg_o = sitk.GetImageFromArray(out_fg_tmp)
            out_seg_o = sitk.GetImageFromArray(out_seg_tmp)
            overlap_seg_o = sitk.GetImageFromArray(overlap_seg_tmp)
            # copy info
            same_dim = False if DATASET_CONFIG[DOMAIN]["method"] == "slic" or "RGB" in DOMAIN else True
            out_fg_o = copy_info(im_obj, out_fg_o, same_dim)
            out_seg_o = copy_info(im_obj, out_seg_o, same_dim)
            overlap_seg_o = copy_info(im_obj, overlap_seg_o, same_dim)
            # create dir paths
            if not os.path.isdir(f'{out_dir}/{stud}/'): os.makedirs(f'{out_dir}/{stud}/')
            seg_fid = os.path.join(out_dir, f'{stud}/{superpix}.nii.gz')
            overlap_fid = os.path.join(out_dir, f'{stud}/{overlappix}.nii.gz')
            msk_fid = os.path.join(out_dir, f'{stud}/{fgmask}.nii.gz')
            # write
            sitk.WriteImage(out_fg_o, msk_fid)
            sitk.WriteImage(out_seg_o, seg_fid)
            sitk.WriteImage(overlap_seg_o, overlap_fid)


def get_n_elements_from_studies(studies):
    dictelems, elem, index = {}, 0, 0
    for stud in studies:
        if stud not in dictelems.keys(): dictelems[stud] = {}
        if DOMAIN == "SUS2020":
            pmfold = glob.glob(f'./SUS2020/niis/study_{stud}/*')
            pmfold = sorted(pmfold, key=lambda x: int(os.path.basename(x.split(".nii.gz")[0])))
        elif "SUS2020_PMs" in DOMAIN: pmfold = glob.glob(f'/home/stud/lucat/PhD_Project/Self-supervised-Fewshot-Medical-Image-Segmentation/data/SUS2020/PMs_niis/{stud}/*')
        for t, img_fid in enumerate(pmfold):
            pm_key = os.path.basename(img_fid).split(".nii.gz")[0]
            if pm_key not in ["TTP", "TMAX", "MIP", "CBF", "CBV"]: continue
            slices = sitk.GetArrayFromImage(sitk.ReadImage(img_fid)).shape[0]
            elem+=slices
            for s in range(slices):
                dictelems[stud][s] = index
                index+=1
    return dictelems, elem


orig_out_dir = out_dir
orig_out_png_dir = out_png_dir

whattouse = compactness if DATASET_CONFIG[DOMAIN]["method"] == "slice" else min_size

for what in whattouse:
    MODE =""
    if DATASET_CONFIG[DOMAIN]["method"]=="felzenszwalb": MODE = DATASET_CONFIG[DOMAIN]["method"]
    elif DATASET_CONFIG[DOMAIN]["method"]=="felzenszwalb_3d": MODE = "3D_felzenszwalb_"+str(what)
    elif DATASET_CONFIG[DOMAIN]["method"]=="slic":
        add = "" if not slic_zero else "_SLICO"
        MODE = str(what) + add  # minimum size of pesudolabels.
    class_flag = ""
    if "penumbra" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["penumbra"]: class_flag = "_PENUMBRA"
    elif "core" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["core"]: class_flag = "_CORE"
    iou_filename = os.path.join(orig_out_png_dir.replace("/placeholder/", ""), f"iou_{MODE}{class_flag}.pkl")
    labels_filename = os.path.join(orig_out_png_dir.replace("/placeholder/", ""), f"nlab_{MODE}{class_flag}.pkl")
    iou_dict_shared, labels_dict_shared = {}, {}
    iou_dict, labels_dict = {}, {}
    # Generate pseudolabels for every image and save them
    s = time.time()
    dictelems, howmanyelems = get_n_elements_from_studies(studies)
    print(round(time.time()-s,2))

    for pm in pmlist:
        print(pm)
        print("-"*50)

        if pm not in iou_dict_shared.keys():
            iou_dict[pm] = np.empty(howmanyelems, dtype=float)
            iou_dict_shared[pm] = Array("d", howmanyelems)
        if pm not in labels_dict_shared.keys():
            labels_dict[pm] = np.empty(howmanyelems, dtype=float)
            labels_dict_shared[pm] = Array("d", howmanyelems)

        out_dir = orig_out_dir.replace("placeholder",pm)
        out_png_dir = orig_out_png_dir.replace("placeholder",pm)
        if "mask_fg" not in DATASET_CONFIG[DOMAIN].keys():
            out_dir = out_dir.replace(pm, pm+"_ALL")
            out_png_dir = out_png_dir.replace(pm, pm+"_ALL")
        else:
            if "mask_ctp" in DATASET_CONFIG[DOMAIN].keys():
                if DATASET_CONFIG[DOMAIN]["mask_ctp"]:
                    add = "_CTP"
                    if "penumbra" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["penumbra"]: add += "_PENUMBRA"
                    elif "core" in DATASET_CONFIG[DOMAIN].keys() and DATASET_CONFIG[DOMAIN]["core"]: add += "_CORE"
                    out_dir = out_dir.replace(pm, pm + add)
                    out_png_dir = out_png_dir.replace(pm, pm + add)
                else:
                    out_dir = out_dir.replace(pm, pm + "_DWI")
                    out_png_dir = out_png_dir.replace(pm, pm + "_DWI")

        with multiprocessing.Pool(processes=16) as pool:
            pool.starmap(single_process, list(zip(studies, [what] * len(studies))))

    for k in iou_dict_shared.keys(): iou_dict[k] = np.frombuffer(iou_dict_shared[k].get_obj())
    for k in labels_dict_shared.keys(): labels_dict[k] = np.frombuffer(labels_dict_shared[k].get_obj())
    if "mask_fg" in DATASET_CONFIG[DOMAIN].keys():
        print(f"Saving {iou_filename} & {labels_filename}...")
        with open(iou_filename, "wb") as f: pkl.dump(iou_dict, f)
        with open(labels_filename, "wb") as ff: pkl.dump(labels_dict, ff)
