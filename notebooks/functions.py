import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import sys

def get_dataset(PATH_DATASET = '../data/DATASET_FINAL.txt'):

    df_dataset = pd.read_csv(PATH_DATASET,delimiter='\t', encoding='utf-8').drop(columns='Batch')
    names = ['X' + str(x) for x in range(len(df_dataset.columns))]

    df_column_name = pd.DataFrame()
    df_column_name['columns'] = names
    df_column_name['names'] = df_dataset.columns  

    df_column_name.to_csv('../data/columns_name.csv',sep=';', encoding='utf-8')

    df_dataset.columns = names

    return(df_dataset,df_column_name)

def miss_col(df_dataset, df_column_name, max_rate_per_col):

    missval = []
    for col in df_dataset.columns:
        missval.append(float(len(np.where(df_dataset[col].astype(str) == 'nan')[0])))

    df_missing = pd.DataFrame()
    df_missing['column'] = df_dataset.columns
    df_missing['count'] = missval
    df_missing['rate (%)'] = 100*np.asarray(missval)/len(df_dataset)
    
    df_dataset = df_dataset[df_missing.loc[df_missing['rate (%)'] <= max_rate_per_col]['column'].tolist()]
            
    missing_columns=[]
    material_description=[]
    for col in df_dataset.columns:
        idx = df_column_name['columns'].tolist().index(col)
        missing_columns.append(col)
        material_description.append(df_column_name['names'].tolist()[idx])
    
    A = df_dataset.as_matrix()
    row, col = A.shape
    missing_image = np.copy(A)
    
    list_count = []
    for j in range(col):
        count = 0.0
        for i in range(row):
            if str(A[i,j]) == 'nan':
                count += 1
                missing_image[i,j] = 0
            else:
                missing_image[i,j] = 255

        list_count.append(count)
    rate = round(100*np.sum(list_count)/(row*col),2)
    
    df_missing = pd.DataFrame()
    df_missing['column'] = missing_columns
    df_missing['material_description|analyte'] = material_description
    df_missing['count missing'] = list_count
    df_missing['rate'] = np.asarray(list_count)/len(df_dataset)

    return(df_dataset, df_missing, missing_image, rate)

def miss_row(df_dataset, df_column_name, max_rate_per_row):

    missval = []
    for i in range(len(df_dataset)):
        missval.append(float(len(np.where(df_dataset.loc[i].astype(str) == 'nan')[0])))

    df_missing = pd.DataFrame()
    df_missing['row'] = np.arange(len(missval))
    df_missing['count'] = missval
    df_missing['rate (%)'] = 100*np.asarray(missval)/len(df_dataset.columns)

    df_dataset = df_dataset.loc[df_missing.loc[df_missing['rate (%)'] <= max_rate_per_row]['row']].reset_index(drop=True)
    
    missing_columns=[]
    material_description=[]
    for col in df_dataset.columns:
        idx = np.where(df_column_name['columns']==col)[0][0]
        missing_columns.append(col)
        material_description.append(df_column_name['names'].tolist()[idx])
    
    A = df_dataset.as_matrix()
    row, col = A.shape
    missing_image = np.copy(A)
    
    list_count = []
    for j in range(col):
        count = 0.0
        for i in range(row):
            if str(A[i,j]) == 'nan':
                count += 1
                missing_image[i,j] = 0
            else:
                missing_image[i,j] = 1

        list_count.append(count)
    rate = round(100*np.sum(list_count)/(row*col),2)
        
    df_missing = pd.DataFrame()
    df_missing['column'] = missing_columns
    df_missing['material_description|analyte'] = material_description
    df_missing['count missing'] = list_count
    df_missing['rate'] = np.asarray(list_count)/len(df_dataset)

    return(df_dataset, df_missing, missing_image, rate)

def combine(df_dataset, col1, col2, col3):

    vect1 = np.copy(df_dataset[col1])
    vect2 = np.copy(df_dataset[col2])

    vect1[np.where(df_dataset[col1].astype(str) == 'nan')[0]] = vect2[np.where(df_dataset[col1].astype(str) == 'nan')[0]] 
    
    df_dataset = df_dataset.drop(columns = [col1,col2])

    df_dataset[col3] = vect1

    return(df_dataset)

def correlation(df_dataset, column, method='pearson'):
    
    PATH_COLNAME = '../data/columns_name.csv'
    df_column_name = pd.read_csv(PATH_COLNAME,delimiter=';', encoding='utf-8')
    index =[] 
    for col in df_dataset.columns:
        if len(np.where(df_column_name['columns']==col)[0]):
            index.append(np.where(df_column_name['columns']==col)[0][0]) 
    df_column_name = df_column_name.loc[index] 

    corr = df_dataset.corr(method=method)

    df_corr = pd.DataFrame()    
    df_corr['attribute'] = corr.columns
    df_corr['correlation'] = corr[column].tolist()
    
    return(corr, df_corr)
