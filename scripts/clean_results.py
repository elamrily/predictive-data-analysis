import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import utils 
import os
import shutil
import sys

reload(sys)
sys.setdefaultencoding('utf8')

PATH_GEN = '../data/genealogy.csv'
PATH_RES = '../data/Results.csv'

# load material code description:
df_mat_desc = utils.get_matcode_desc(PATH_GEN)

# load results table and mean results per batch:
df_res, df_mean_res = utils.get_res(PATH_RES, PATH_GEN)

if __name__=='__main__':

    df_clean = pd.DataFrame()
    count = 0

    # browse all material codes:
    for material_code in np.unique(df_res['MATCODE']):

        print('-------------------------------------------------------------------------------')
        
        # create a temporary dataframe which contains all results of the material code:
        idx = np.where(df_res['MATCODE'] == material_code)[0]
        ind = np.where(df_mat_desc['material_code'] == material_code)[0]
        df_temp = pd.DataFrame(df_res.as_matrix()[idx],columns=df_res.columns)

        # browse all analytes of the material code:
        for analyte in np.unique(df_temp['ANALYTE']):
            count+=1
            print('-------------------------------------------------------------------------------')
            print('component : ',df_mat_desc['material_desc'][ind].tolist()[0], ' : ',material_code, '\t | ','analyte : ',analyte.encode('utf-8'))

            # create a temporary dataframe which contains all results of one analyte of the material code:
            idx = np.where(df_temp['ANALYTE'] == analyte.encode('utf-8'))[0]
            df_temp1 = pd.DataFrame(df_temp.as_matrix()[idx],columns=df_temp.columns)

            # necessary informations for data cleaning:
            nb_points = float(len(df_temp1['FINAL']))
            Low_tuck,Upp_tuck,mean,std = utils.summary(df_temp1['FINAL'].tolist())

            # clean data:
            df_out_sup = df_temp1[df_temp1.FINAL > (mean + 5*std)]
            df_out_inf = df_temp1[df_temp1.FINAL < (mean - 5*std)]
            df_out = pd.concat([df_out_sup,df_out_inf])
            df_concat = pd.concat([df_temp1,df_out])

            # some statistics
            nb_outliers = float(len(df_out))
            rate = round(float((nb_outliers/nb_points))*100,3)
            df_temp1 = df_concat.drop_duplicates(keep = False)
            nb_new_data = float(len(df_temp1))
            print('|Nb_points   : ',nb_points)
            print('|Nb_outliers : ',nb_outliers)
            print('|Nb_new_data : ',nb_new_data)
            print('|Rate        : ',rate)
            print('\n\t\t\t\t\t-',count,'-\n')
            
            # target variables must not be cleaned:
            df_temp4 = pd.DataFrame(df_temp1)
            if analyte == 'Side streaming max' or analyte == 'Side streaming (maxi)':
                df_temp1 = df_temp4
            df_clean = pd.concat([df_clean,df_temp1])

    # save cleaned results:
    df_clean.to_csv('../data/Results_clean.csv',encoding='utf-8',sep=';')