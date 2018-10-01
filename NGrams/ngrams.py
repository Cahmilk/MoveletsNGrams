import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import time
from joblib import Parallel, delayed
import multiprocessing
import threading
import random

def read_csv(csv_name):
    return pd.read_csv(csv_name, sep=',')


def dict_to_csv(dict1, dict2):
    
    df1 = pd.DataFrame.from_dict(dict1)
    df2 = pd.DataFrame.from_dict(dict2)

    df1_label = df1['label'];
    df1.drop('label', axis=1, inplace=True);
    df1 = pd.concat([df1, df1_label], axis=1);

    df2_label = df2['label'];
    df2.drop('label', axis=1, inplace=True);
    df2 = pd.concat([df2, df2_label], axis=1);

    df1.to_csv('D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\Movelets\\newfour8_ED\\2grams_test.csv', index=False)
    df2.to_csv('D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\Movelets\\newfour8_ED\\2grams_train.csv', index=False)


def set_dict(df):
    return df.to_dict(orient='list')

df_test = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\Movelets\\newfour8_ED\\test_histogram.csv") 
df_train = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\Movelets\\newfour8_ED\\train_histogram.csv")
#df_test = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\12_week\\MoveletsSize10WithDefPrunning\\Movelets\\newfour8_ED\\freq_test.csv") 
#df_train = read_csv("D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\12_week\\MoveletsSize10WithDefPrunning\\Movelets\\newfour8_ED\\freq_train.csv")

test_dict = set_dict(df_test)
train_dict = set_dict(df_train)

def drop_indices(df):
    return df.drop('0', axis=1)

def create_Series():
    return pd.Series()

def set_series(list_name):
    new_series = pd.Series(list_name)
    #new_series = new_series.rename(name)
    return new_series

def compare_freq(val1, val2):

    if(val1>0 and val2>0):
        return (max(val1,val2))
    else:
        return 0

def run_rows(df, f, s):

    n_col = []
    for index,row in df.iterrows():
        n_col.append(compare_freq(row[f], row[s]))

    return n_col

def get_label(df):
    return df.iloc[:, -1]

def drop_label(df):
    return df.drop(['label'], axis=1)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def add_new_column(f, s, n_col, df_dict):
    name = str(f) + "+" + str(s)
    new_column = n_col
    df_dict[name] = new_column
    return df_dict

def test(df1, df2, s, f):

    global train_dict, test_dict
    #Take frequence relation for the train rows.
    n_col = run_rows(df1, s, f)
    test_dict_aux = dict()
    train_dict_aux = dict()
    new_dict = dict()

    #If its frequences mutually happened at least once, create new column to the data frame, and calculate the frequency for the test set
    if(np.count_nonzero(n_col)>0): 

        n_col_test = run_rows(df2, f, s) 

        train_dict_aux = add_new_column(f, s, n_col, train_dict_aux)
        test_dict_aux = add_new_column(f, s, n_col_test, test_dict_aux)
        new_dict['train'] = train_dict_aux
        new_dict['test'] = test_dict_aux
        return new_dict

def shuffle_list(lista):
    random.seed(1)
    random.shuffle(lista)
    return lista

def get_shuffled_movelets(df1):
    lista = shuffle_list(list(df1.columns.values))
    #return lista
    return lista[:int(len(lista)/2)]


def two_grams(df1, df2):

    global train_dict, test_dict

    movelets1 = get_shuffled_movelets(df1)
    movelets2 =  get_shuffled_movelets(df1)

    #It will select the first movelet for comparison
    for f in movelets1:
        print(len(movelets2))
        movelets2.remove(f)

        if(len(movelets2)>0):
            new_aux = Parallel(n_jobs=6)(delayed(test)(df1, df2, s, f) for s in movelets2)
            
            for el in new_aux:
                if el != None:
                    for key, value in el.items():
                        if(key=='test'):
                            test_dict = merge_two_dicts(test_dict, value)
                        elif (key=='train'):
                            train_dict = merge_two_dicts(train_dict, value)

#Get label from dataframes
df_train_label = (get_label(df_train)).tolist() 
df_test_label = (get_label(df_test)).tolist() 

#take off the labels from data frames
df_train = drop_label(df_train)
df_test = drop_label(df_test)

two_grams(df_train, df_test)

train_dict["label"] = df_train_label
test_dict["label"] = df_test_label

dict_to_csv(train_dict, test_dict)