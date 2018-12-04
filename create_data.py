"""Get X.npy and Y.npy, which will be used for training the model"""
import numpy as np
import glob
import os, sys

#Get AutisticX.npy
fpath = r"AutisticX.npy"
npyfilespath = r"Autistic/"
os.chdir(npyfilespath)
npfiles = glob.glob("*.npy")
all_arrays = []
for i, npfile in enumerate(npfiles):
    x = np.load(os.path.join(npyfilespath, npfile))
    x = np.triu(x,0)
    x = x.flatten()
    x = x[x!=0]
    all_arrays.append(x)
all_arrays = np.array(all_arrays)
all_arrays.shape
np.save(fpath, all_arrays)

#Get NormalX.npy
fpath = r"NormalX.npy"
npyfilespath = r"Normal/"
os.chdir(npyfilespath)
npfiles = glob.glob("*.npy")
all_arrays = []
for i, npfile in enumerate(npfiles):
    x = np.load(os.path.join(npyfilespath, npfile))
    x = np.triu(x,0)
    x = x.flatten()
    x = x[x!=0]
    all_arrays.append(x)
all_arrays = np.array(all_arrays)
all_arrays.shape
np.save(fpath, all_arrays)

#Get X.npy
X_autistic = np.load(r"AutisticX.npy")
X_normal = np.load(r"NormalX.npy")
np.save(r"X.npy", np.vstack((X_autistic, X_normal)))

#Get Y.npy
Y_autistic = np.ones(387)
Y_normal = np.zeros(436)
Y = np.concatenate((Y_autistic, Y_normal),axis=0)
np.save(r"Y.npy", Y)
