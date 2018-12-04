"""Create PSMs and PSM matrices including the summed PSM/PSM Matrix"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import h5py

num_Of_PSMs = 1 #Get the first n psms
summed_psm = np.zeros((34716,)) #Stored the summed psms

with h5py.File('/data/jang047/eigenvectors.hdf5', 'r') as g:
        print('reading eigenvectors')
	eigenvectors = g['eigenvectors']
	eigenvectors = np.array(eigenvectors)
	eigenvectors = np.real(eigenvectors) #Remove imaginary part of vectors

for i in range(num_Of_PSMs):
        with h5py.File('PSM_'+str(i)+'.hdf5', 'w') as f: #Save each individual PSM
            print("saving"+str(i))
            dset = f.create_dataset('PSM_'+str(i), data=eigenvectors[:][i])
            psm = pd.DataFrame(eigenvectors[:][i])
            psm.to_csv('PSM_'+str(i)+'.csv')
            summed_psm = np.add(summed_psm, eigenvectors[:][i], out=summed_psm, casting='unsafe')

#Save summed PSM
with h5py.File('PSM_SUM.hdf5', 'w') as h:
        print("saving summed psm")
        dset = h.create_dataset('PSM_SUM', data=summed_psm)
        summed_psm = pd.DataFrame(summed_psm)
        psm.to_csv('PSM_SUM.csv')

#Open PSM and recreate PSM matrix, then save it
for i in range(num_Of_PSMs):
        with h5py.File('PSM_'+str(i)+'.hdf5', 'r') as g:
                PSM = g['PSM_'+str(i)]
                PSM = np.array(PSM)
                matrix = np.ndarray(shape=(264,264), dtype=float) #matrix stores final desired result
                idx = np.triu_indices(264, k=1) #Get indices of upper right triangle of matrix, excluding the diagonal
                matrix[:] = 0 #Set all values to 0
                matrix[idx] = PSM #Place ssm values into upper right triangle of matrix, using the indices defined above
                idx = np.tril_indices(264, k=-1) #Get indices of lower left triangle, excluding the diagonal
                matrix[idx] = matrix.T[idx] #Copy the mirror image of upper triangle to lower triangle
        with h5py.File('PSM_matrix_'+str(i)+'.hdf5', 'w') as f:
                dset = f.create_dataset('PSM_matrix_'+str(i), data=matrix)
                matrix = pd.DataFrame(matrix)
                matrix.to_csv('PSM_matrix_'+str(i)+'.csv')

#Open summed PSM and recreate summed PSM matrix
with h5py.File('PSM_SUM.hdf5', 'r') as g:
        PSM = g['PSM_SUM']
        PSM = np.array(PSM)
        matrix = np.ndarray(shape=(264,264), dtype=float) #matrix stores final desired result
        idx = np.triu_indices(264, k=1) #Get indices of upper right triangle of matrix, excluding the diagonal
        matrix[:] = 0 #Set all values to 0
        matrix[idx] = PSM #Place ssm values into upper right triangle of matrix, using the indices defined above
        idx = np.tril_indices(264, k=-1) #Get indices of lower left triangle, excluding the diagonal
        matrix[idx] = matrix.T[idx] #Copy the mirror image of upper triangle to lower triangle

#Save summed PSM matrix
with h5py.File('PSM_SUM_matrix.hdf5', 'w') as h:
        print("saving summed psm matrix")
        dset = h.create_dataset('PSM_SUM_matrix', data=matrix)
        matrix = pd.DataFrame(matrix)
        psm.to_csv('PSM_SUM_matrix.csv')
