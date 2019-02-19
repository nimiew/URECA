"""Moves data(npy files) into Autistic or Normal folderS"""
import pandas as pd
import glob
import shutil
#Create a dictionary that maps filename to 1(Autistic) or 2(Normal)
df = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")
df = df.filter(items=['FILE_ID', 'DX_GROUP'])
df = df[df['FILE_ID'] != "no_filename"]
full_dict = dict(zip(df['FILE_ID'],df['DX_GROUP']))
#Get all the filenames of data
mylist = [f for f in glob.glob("*.npy")]
#Check if each data belongs to Autistic or Normal and move it to the corresponding folder
for i in range(len(mylist)):
    if full_dict.get(mylist[i][:-10]) == 1:
        shutil.move(mylist[i], "Autistic/")
    elif full_dict.get(mylist[i][:-10]) == 2:
        shutil.move(mylist[i], "Normal/")
