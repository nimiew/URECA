"""The original phenotype csv had records where FILE_ID = 'no_filename' but upon closer inspection, we can still get the url to download it, by editting the phenotype filename"""
import pandas as pd

df = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")

for i in range(len(df)):
    if (df['FILE_ID'][i] == 'no_filename'): #Find records where FILE_ID = 'no_filename'
        if(i+1!=len(df) and df['FILE_ID'][i+1] != 'no_filename' and df['SITE_ID'][i] == df['SITE_ID'][i+1]): #Check if we can use the tuple in front to generate filename
            df['FILE_ID'][i] = df['FILE_ID'][i+1][:-5]+str(df.iloc[i]['subject'])
        elif(i-1!=-1 and df['FILE_ID'][i-1] != 'no_filename' and df['SITE_ID'][i] == df['SITE_ID'][i-11]): #Check if we can use the tuple in front to generate filename
            df['FILE_ID'][i] = df['FILE_ID'][i-1][:-5]+str(df.iloc[i]['subject'])
        else:
            print('Failed to change filename properly')

df.to_csv("Phenotypic_V1_0b_preprocessed1_fixed_filesnames.csv")
