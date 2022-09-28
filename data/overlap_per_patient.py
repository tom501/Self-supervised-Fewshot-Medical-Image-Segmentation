import matplotlib.pyplot as plt
import os, glob, PIL, multiprocessing
import numpy as np
import pickle as pkl

from skimage.measure import label
from sklearn.metrics import jaccard_score

root = "/home/stud/lucat/PhD_Project/Self-supervised-Fewshot-Medical-Image-Segmentation/data/SUS2020/"
folds = ["3D-FELZENSZWALB_PMs_stacked_RGB_pngs","FELZENSZWALB_PMs_stacked_RGB_pngs","FELZENSZWALB_PMs_RGB_pngs",
         "SLIC_PMs_RGB_pngs","SLIC_PMs_stacked_RGB_pngs"]


def single_fold(f):
    values = {}
    print(f)
    if f not in values.keys(): values[f] = {}
    comppath = os.path.join(root, f)
    for exp_fold in glob.glob(comppath+"/*"):
        if not os.path.isdir(exp_fold): continue
        print(exp_fold)
        exp = exp_fold.replace(comppath+"/","")
        if exp not in values[f].keys(): values[f][exp] = {}
        for p_fold in glob.glob(exp_fold+"/*"):
            patient = p_fold.replace(exp_fold+"/","")
            mask_list = glob.glob(p_fold+"/fgmask_*")
            set_typeexp = {}
            curr_exp = {}
            for maskpath in sorted(mask_list):
                idx = maskpath.split("_")[-1]
                overlap_list = glob.glob(p_fold+"/overlap*"+idx)

                mask_img = np.asarray(PIL.Image.open(maskpath), dtype=np.float32)
                if len(np.unique(mask_img))==1: continue
                for overlap in overlap_list:
                    typeexp = overlap.replace(p_fold+"/","").split(idx)[0][:-1]
                    if typeexp not in set_typeexp.keys(): set_typeexp[typeexp] = 0
                    if typeexp not in curr_exp.keys(): curr_exp[typeexp] = []
                    if typeexp not in values[f][exp].keys(): values[f][exp][typeexp] = {"x":[], "y":[]}
                    if patient not in values[f][exp][typeexp]["x"]: values[f][exp][typeexp]["x"].append(patient)

                    overlap_img = np.asarray(PIL.Image.open(overlap), dtype=np.float32)
                    largestcc = getLargestCC(overlap_img)
                    overlap_img = np.array(largestcc * 255, dtype=np.float32)
                    #overlap_img = np.array((overlap_img>0)*255, dtype=np.float32)
                    jacc_score = jaccard_score(mask_img, overlap_img, average="micro", zero_division=0.0)
                    curr_exp[typeexp].append(jacc_score)
            for typeexp in set_typeexp.keys():
                mean_val = round(np.mean(curr_exp[typeexp]),4)
                values[f][exp][typeexp]["y"].append(mean_val)

    with open(root+"largestCC/_"+f+"_overlap_patients.pkl", "wb") as file: pkl.dump(values, file)


def getLargestCC(segmentation):  # largest connected components
    labels = label(segmentation)
    #assert (labels.max() != 0)  # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else: largestCC = np.zeros(segmentation.shape)
    return largestCC


with multiprocessing.Pool(processes=len(folds)) as pool:
    pool.map(single_fold,folds)
