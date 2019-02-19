1.	Run download_abide_preproc.py
	Use:
	nohup python download_abide_preproc.py -d func_preproc -p cpac -s filt_global -o <Storage_location>
	
2.	Run calculate_corr_ABIDE.py to get npy files

3.	Run classify_file.py to create 2 folders, and place npy files into the corresponding folder(austistic or normal)

4.	Run create_data.py to obtain X.npy and Y.npy for training

5.	Train using feed_forward_net.py and iterate values of parameters to find best values for parameters

6.	Compare results with other models, by running SVM.py

7.	With the best values of parameters, run save_model.py to save the model

8.	Run create_SSM.py to produce SSM(including matrix), stored in both CSV and hdf5 formats

9.	Run create_K.py to produce K, stored in hdf5 format

10.	Run create_eigen.py to produce eigenvalues(stored in both CSV and hdf5) and eigenvectors(stored in hdf5)

11.	Run new_PSM.py to generate tables for important eigenvectors.
	


