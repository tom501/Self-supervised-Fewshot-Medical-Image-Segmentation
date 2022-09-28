import os, glob, cv2
from numpy import binary_repr

root_path = "/home/stud/lucat/CTP_01_001_v2.0/"
gt_folder = root_path + "Ground truth/"

ids = os.listdir(root_path)

for curr_id in ids:
    if curr_id == "Ground truth": continue
    suffix = ".png"
    if curr_id == "raw_input": suffix = ".tiff"

    sv_img = glob.glob(f'{root_path}{curr_id}/*')

    n_sv = curr_id.split("_")[0]
    what = curr_id.split("_")[-1]
    toadd = "overlap" if what == "overlap" else "superpix"
    fileprefix = toadd + "-3D_felzenszwalb_" + n_sv + "_"
    sv_img = sorted(sv_img, key=lambda x: int(os.path.basename(x).split(fileprefix)[-1].replace(suffix, "")))
    print(sv_img)

    for imgname in sv_img:
        gt_idx = gt_folder + "fgmask_" + os.path.basename(imgname).split(fileprefix)[-1]
        if curr_id == "raw_input": gt_idx = gt_idx.replace(suffix, ".png")
        gt = cv2.imread(gt_idx, cv2.IMREAD_GRAYSCALE)
        if "TMAX" in imgname or "CBV" in imgname or "TTP" in imgname or "CBF" in imgname: img = cv2.imread(imgname)
        else:
            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)
            
        _, roi = cv2.threshold(gt, 200, 255, cv2.THRESH_BINARY)
        binary_img = (img>0)*1
        binary_img = binary_img[:,:,0]
        cnt_mask, _ = cv2.findContours(binary_img.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, cnt_mask, -1, (0, 255, 0), 1)
        cnt, _ = cv2.findContours(roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, cnt, -1, (255, 0, 0), 2)

        cv2.imwrite(imgname, img)
