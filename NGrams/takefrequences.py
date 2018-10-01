import csv
import pandas as pd
import tensorflow as tf
import os


def read_csv(csv_name):
	return pd.read_csv(csv_name, sep=',')

def get_keys(df):
	return df.keys()

def get_files_in_dir(dirname):
    return os.listdir(dirname)

dir_name = "D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\12_week\\MoveletsSize10WithDefPrunning\\Movelets\\newfour8_ED"
files_in_dir = get_files_in_dir(dir_name)

files_in_dir = files_in_dir[:-4]

files = dir_name + '\\' + files_in_dir[0];
archs_in_file = os.listdir(files);

if(len(archs_in_file) > 1):
	
	files_aux = files + "\\" + archs_in_file[5];

	#This "if" takes dataframes filled up with the movelets.
	if(os.path.isfile(files_aux)):
		df = read_csv(files_aux)

files_in_dir.pop(0);

for files in files_in_dir:

	files = dir_name + '\\' + files;
	archs_in_file = os.listdir(files)

	if(len(archs_in_file) > 1):

		files_aux = files + "\\" + archs_in_file[5];

		#This "if" takes dataframes filled up with the movelets.
		if(os.path.isfile(files_aux)):
			
			df_aux = read_csv(files_aux);
			df_aux.drop('label', axis=1, inplace=True);
			df = pd.concat([df, df_aux], axis=1);

df_label = df['label'];
df.drop('label', axis=1, inplace=True);

df = pd.concat([df, df_label], axis=1);

df.to_csv(dir_name +'\\freq_train.csv', index=False);