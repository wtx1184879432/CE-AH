import scipy.io as sio
import numpy as np
import os
import nibabel as nib

template_cors = sio.loadmat('cors.mat')
template_cors = template_cors['patch_centers']

# 建立块坐标与编号的字典
dic = {}
u_1 = 0
for i in range(12,187,25):
    for j in range(12,212,25):
        for k in range(12, 187, 25):
            list = [i, j, k]
            unit = {u_1:list}
            dic.update(unit)
            u_1 = u_1 + 1

# 建立21-80块的编号与坐标字典
dic_1 = {}
for i in range(0,80):
    list = [template_cors[0][i],template_cors[1][i],template_cors[2][i]]
    for v in dic.values():
        if(list == v):
            key = [k for k, v in dic.items() if v == list]
            unit = {key[0]:list}
            dic_1.update(unit)

data = sio.loadmat('data_1.mat')

image_files = data['samples_train'].flatten()

# 建立1-80块的编号与体素值不为0的个数字典
dic_2 = {}
u = 0
for v in dic_1.values():
    i = v[0]
    j = v[1]
    k = v[2]
    key = [k for k, v_1 in dic.items() if v_1 == v]
    for file in image_files:
        img_dir = os.path.join('./ADNI', file)
        img_dir = img_dir.replace(".npy", ".nii").rstrip()

        img = nib.load(img_dir).get_fdata()[:181, :217, :181]
        img_patch = img[i - 12: i + 12 + 1,
                    j - 12: j + 12 + 1,
                    k - 12: k + 12 + 1]
        for a in range(0, 25):
            for b in range(0, 25):
                for c in range(0, 25):
                    if img_patch[a, b, c] != 0:
                        u = u + 1

    u = u / 547
    print(u)
    unit = {key[0]: u}
    dic_2.update(unit)
    u = 0

a = sorted(dic_2.items(), key=lambda x: x[1])
# u值升序的块号字典
dic_3 = dict(a)
dic_4 = [[k, v] for k, v in dic_3.items()]
dic_4 = np.array(dic_4)
sio.savemat('u_value_up.mat', {"block": dic_4})