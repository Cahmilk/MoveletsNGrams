import pandas as pd
import tensorflow as tf
import numpy as np
import keras

def read_csv(csv_name):
	return pd.read_csv(csv_name, sep=',', header=None)

df_test = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\MoveletsSize10WithPrunning\\Movelets\\newfour8_ED\\freq_test.csv") 
df_train = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\MoveletsSize10WithPrunning\\Movelets\\newfour8_ED\\freq_train.csv") 


header_list = list(df_test.columns.values)

header_aux = list()

del header_list[-1]

print(header_list)
for col in header_list:

    hg = 0;

    for el in df_test[col]:

        if isinstance(el, int):

            if(el>hg):
                hg = el

    if(hg<1):
        header_aux.append(col);


df_test = df_test.drop(header_aux, axis=1)
df_train = df_train.drop(header_aux, axis=1)

df_test.to_csv('D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\MoveletsSize10WithPrunning\\Movelets\\newfour8_ED\\new_freq_test.csv', header=False)
df_train.to_csv('D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\MoveletsSize10WithPrunning\\Movelets\\newfour8_ED\\new_freq_train.csv', header=False)

#Take the first column of data frame, and iterate over the rest of the columns. Drop the first column. Make the second become the first. And so on.
#header_first = list(df_test.columns.values)
#header_second = header_first.pop(0)

#print(header_first)


#
#print(df.count())