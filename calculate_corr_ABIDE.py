
"""
Extract Timeseries from 264 sphere nodes of Power Atlas
Radias:  2.5 (5mm radius) -> in accord with (Power et al. 2011)

Input: 
    .npy file - rsfMRI data
    .txt file - MNI coordinates for Power Atlas

Output:
    S*_Sphere_Timeseries.mat
    S*_Sphere_Corre.mat
    ROI_measures
           1. SNR
           2. ReHo
           3. mcorr
          
Xiuchao.Sui@gmail.com                      
"""

import pdb
import os
import re
import numpy as np
from scipy.stats import pearsonr
from scipy.io    import savemat
from scipy import  linalg
import itertools
import nibabel as nib
from multiprocessing import Pool
from nibabel.affines import apply_affine
import numpy.linalg as npl
 
def partialCorr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
 
 
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
 
 
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
     
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
 
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
             
            corr = pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
         
    return P_corr

def convertMNItoIJK(affine, mni_coord): 
    ijk = apply_affine(npl.inv(affine), mni_coord)
    ijk = np.rint(ijk)
    ijk_int = ijk.astype(int)
    return ijk_int

def setParameters():
    config = { 'R': 2.5}
    # usage
    #print ("Sphere radius is %.1f mm" %(config['R']))
    return config
    
def setPowerSeeds(power_filename):
    POWER = open( power_filename)
    PowerMNI = []
    for line in POWER:
        line = line.strip()
        coords = re.split(" +", line)
        coords = [int(x) for x in coords]
        PowerMNI.append(coords)
    
    PowerMNI = np.array(PowerMNI)
    K = len(PowerMNI)
    print ("%d Power seeds read" %K)
    return PowerMNI

def isValidVoxel(matrix, v):
    if np.count_nonzero( (v < 0) | (v >= matrix.shape[:3]) ) > 0:
        return False
    if np.count_nonzero(matrix[tuple(v)]) == 0: #the time series contains all zeroes
        return False
    return True

def genBall(R):
    R2 = int(R)
    dv_inBall = []
    var_range = range( -R2, R2 + 1 );
    for dx, dy, dz in itertools.product(var_range, var_range, var_range):
        if dx*dx + dy*dy + dz*dz <= R*R:
            dv_inBall.append( [ dx, dy, dz ] )    
    return np.array(dv_inBall)
           
def getValidCoordsInBall(matrix, v, dv_inBall):
    coords_inBall = dv_inBall + v
    validCoordsInBall = []
    for v2 in coords_inBall:
        if isValidVoxel(matrix, v2):
            validCoordsInBall.append(v2)
    return np.array(validCoordsInBall)

def genBox():
    '''27 voxels cubicle'''
    dv_inBox = []
    var_range = range(-1, 2)  
    for dx, dy, dz in itertools.product(var_range,var_range,var_range):
        dv_inBox.append([dx,dy,dz])
    return np.array(dv_inBox)

def getValidCoordsInBox(matrix, v, dv_inBox):
    voxelsCoords = dv_inBox + v
    validCoordsInBox = []
    for v2 in voxelsCoords:
        if isValidVoxel(matrix, v2):
            validCoordsInBox.append(v2)
    return np.array(validCoordsInBox) 


def ROI_Corr(matrix, ROI_voxs): 
    ts_voxs = np.copy( matrix[ROI_voxs[:,0], ROI_voxs[:,1], ROI_voxs[:,2]])    
    W = np.corrcoef(ts_voxs)
    Wtri = np.triu(W)
    triVec = Wtri[np.where(Wtri!=0)]    
    mcorr = np.mean(triVec)
    return mcorr
    
def kendal(X):
    """X: (n Timepoints, 27voxels) """
    n,k = X.shape
    sr = np.sum(X,1)
    srb = np.mean(sr)
    s = np.sum(sr**2) - n*(srb**2)
    X_kendal = 12*s/k**2/(n**3 - n)
    return X_kendal
    
def ROI_ReHo(matrix, ROI_Voxs):
    ROI_ReHo = []
    for v in ROI_Voxs:
        dv_inBox = genBox()
        BoxCoords = getValidCoordsInBox(matrix, v, dv_inBox)
        ts_box = np.copy( matrix[BoxCoords[:,0], BoxCoords[:,1], BoxCoords[:,2]])
        ReHo = kendal(ts_box.T)
        ROI_ReHo.append(ReHo)
    ROI_ReHo = np.array(ROI_ReHo)     
    mReHo = np.mean(ROI_ReHo)
    return mReHo     
    
def ROI_SNR(matrix, ROI_voxs):
    '''calculate ROI_SNR'''
    ts_voxs = np.copy( matrix[ ROI_voxs[:,0], ROI_voxs[:,1], ROI_voxs[:,2]]) 
    snr = signaltonoise(ts_voxs, axis=1)     
    mSNR = np.mean(snr)  
    return mSNR

def meanTS(matrix, ROI_voxs):
    # pdb.set_trace()
    ts_voxs = np.copy( matrix[ROI_voxs[:, 0], ROI_voxs[:, 1], ROI_voxs[:, 2]] )
    meanTimeseries = np.mean( ts_voxs, axis=0 )
    return meanTimeseries 

def saveResults(config, meanTimeseries, subject):
    # save Corre.mat   
    #print ("Computing correlation matrix between %d regions..." %K)
    #pearsonCorr = partialCorr(np.transpose(meanTimeseries))
    pearsonCorr = np.corrcoef(meanTimeseries)
    np.fill_diagonal(pearsonCorr, 0);
    
    #imgdir = os.path.dirname(config['img_file'])
    #corrMatFilename = os.path.join(imgdir, 'Power', "Sphere_Corre_" + ".mat")
    corrNumpyFile =  os.path.join(save_folder, subject + '_power.npy')
    #savemat(corrMatFilename, { 'Corre': pearsonCorr } )

    try:
        os.mkdir(os.path.dirname(corrNumpyFile))
    except:
        pass
    np.save(corrNumpyFile, pearsonCorr)


	# save Timeseries.mat
    #tsMatFilename = os.path.join(imgdir, 'Power', "Sphere_Timeseries_" + suffix + ".mat")       
    #savemat( tsMatFilename, { 'TS': meanTimeseries } )
     
    #print ("Correlation matrix", pearsonCorr, " saved to '%s'." % corrNumpyFile)
   
def saveROImeasures(config, ROIs_SNR, ROIs_mReHo, ROIs_mcorr, suffix):
    #print("Saving SNR, reho and mcorr of Power Nodes (5mm radius sphere)")
    imgdir = os.path.dirname(config['img_file'])
    #snrMatFile = os.path.join(imgdir, 'Power', "ROI_SNR_" + suffix + ".mat")
    rehoMatFile = os.path.join(imgdir, 'Power', "ROI_reho_" + suffix + ".mat")
    mvoxcorrMatFile = os.path.join(imgdir, 'Power', "ROI_voxcorr_" + suffix + ".mat")
    
    #savemat( snrMatFile, { 'snr': ROIs_SNR } )
    savemat( rehoMatFile, { 'reho': ROIs_mReHo } )
    savemat( mvoxcorrMatFile, { 'corr': ROIs_mcorr } )
    
    '''
    csvMeasures = imgdir + "/Measures_" + afflix + ".csv"
    ofile = open(csvMeasures, "wb")
    writer = csv.writer(ofile, delimiter = ',')
    writer.writerow(ROIs_mcorr)
    ofile.close()
    '''
    return True
    
def PowerSphere(params):
    img_file, power_coords_mni, subject = params
    

    print ("loading file from '%s'..." % (img_file))
    img = nib.load(img_file)
    affine = img.affine
    matrix = img.get_data()

    config = setParameters()
    config['affine'] = affine
    config['img_file'] = img_file
    #print ("Dimensions: %dx%dx%dx%d." % matrix.shape)
    #print ("Affine:", affine)

    K = config['K'] = len(power_coords_mni)
    dv_inBall = genBall(config['R'])

    # initialize 
    ROIs_voxels= [[]for k in range (K)]
    meanTimeseries  = np.zeros((K, matrix.shape[3]))
    power_coords_ijk = np.zeros((K, 3))
    power_coords_ijk = power_coords_ijk.astype(int)
    ROIs_mcorr = np.zeros(K)
    ROIs_mReHo = np.zeros(K)
    ROIs_SNR = np.zeros(K)
        
    for k in range(K):
        power_coords_ijk[k] = convertMNItoIJK(config['affine'], power_coords_mni[k])
        #print ('MNI:', power_coords_mni[k], 'IJK:', power_coords_ijk[k])
        ROIs_voxels[k] = getValidCoordsInBall(matrix, power_coords_ijk[k], dv_inBall)
        #ROIs_voxels[k] = np.array(ROIs_voxels[k])
        
        #print ('For index k:', k, 'Coords: ', power_coords_ijk[k], 'Voxels in Sphere:', len(ROIs_voxels[k]))
        if len(ROIs_voxels[k]) != 0:
            meanTimeseries[k] = meanTS(matrix, ROIs_voxels[k])   

    saveResults(config, meanTimeseries, subject) 

    '''
    for k in range(K):
        ROIs_mcorr[k] = ROI_Corr(matrix, ROIs_voxels[k])
        ROIs_mReHo[k] = ROI_ReHo(matrix, ROIs_voxels[k] )
        #ROIs_SNR[k] = ROI_SNR(matrix, ROIs_voxels[k] )

    saveROImeasures(config, ROIs_SNR, ROIs_mReHo, ROIs_mcorr)  
    '''

    return True

# ---------------------------- Loop Subjects ----------------------------


root = './Outputs/cpac/filt_global/func_preproc/'       
save_folder = 'Processed_ABIDE/'
subject_list = os.listdir(root)
power_filename = 'PowerVoxMNI.txt'
power_coords = np.genfromtxt(power_filename)
K = len(power_coords)

if __name__ == '__main__':
    p = Pool(10)
    params = []
    for subject in subject_list:
        subject = subject[0:-20]

        if os.path.isdir(root):
            img_file = os.path.join(root, subject + '_func_preproc.nii.gz') #timeseries data
            try:
                os.mkdir(save_dir)
            except:
                pass

            if os.path.isfile(img_file):
                print ('------ subject %s file %s-----' %(subject, img_file))
                params.append(tuple([img_file, power_coords, subject]))

    p.map(PowerSphere, params)
    p.close() # no more tasks
    p.join()  # wrap up current tasks
