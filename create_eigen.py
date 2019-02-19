import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import h5py

#Load K
with h5py.File('/data/jang047/K_823_ffn.hdf5', 'r') as f:
    K = f['K']
    K = np.array(K)
    eig_vals, eig_vecs = np.linalg.eig(K)#Calculate eigenvalues and vectors
#Save eigenvalues
with h5py.File('/data/jang047/K_823_ffn_eigenvalues.hdf5', 'w') as f:
    dset = f.create_dataset("eigenvalues", data=eig_vals)
#Save eigenvectors
with h5py.File('/data/jang047/K_823_ffn_eigenvectors.hdf5', 'w') as f:
    dset = f.create_dataset("eigenvectors", data=eig_vecs)
#Remove imaginary numbers of eigen values and save into CSV
with h5py.File('/data/jang047/K_823_ffn_eigenvalues.hdf5', 'r') as g:
    eigenvalues = g['eigenvalues']
    eigenvalues = np.array(eigenvalues)
    eigenvalues = np.real(eigenvalues)
    result = pd.DataFrame(eigenvalues)
    result.to_csv("K_823_ffn_eigenvalues.csv")
