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

    # browse all material codes:
    for material_code in np.unique(df_res['MATCODE']):

        print('-------------------------------------------------------------------------------')

        # create a temporary dataframe which contains all results of the material code:
        idx = np.where(df_res['MATCODE'] == material_code)[0]
        ind = np.where(df_mat_desc['material_code'] == material_code)[0]
        df_temp = pd.DataFrame(df_res.as_matrix()[idx],columns=df_res.columns)

        # replace '/' by '-' in the description name:
        description = str(df_mat_desc['material_desc'][ind].tolist()[0])
        if '/' in description:
            description = description.replace('/','-')

        # folder name setting:
        folder = description + ' : ' + str(material_code)

        # create material description folder if it doesn't exist:
        if folder not in os.listdir('../infoMatcode/'):
            os.mkdir('../infoMatcode/' + folder)

        # browse all analytes of the material code:
        for analyte in np.unique(df_temp['ANALYTE']):

            # create analyte folder if it doesn't exist:
            anal = analyte
            if '/' in str(analyte):
                anal = anal.replace('/','_')

            if str(anal) not in os.listdir('../infoMatcode/'+folder):
                os.mkdir('../infoMatcode/' + folder + '/' + str(anal))

            # create clean and brut folders if it doesn't exist to store scatterplots:
            if 'brut' not in os.listdir('../infoMatcode/'+folder + '/' + str(anal)):
                os.mkdir('../infoMatcode/' + folder + '/' + str(anal) + '/brut')
                os.mkdir('../infoMatcode/' + folder + '/' + str(anal) + '/clean')

            print('-------------------------------------------------------------------------------')
            print('component : ',df_mat_desc['material_desc'][ind].tolist()[0], ' : ',material_code, '\t | ','analyte : ',analyte.encode('utf-8'))

            # create a temporary dataframe which contains all results of one analyte of the material code:
            idx = np.where(df_temp['ANALYTE'] == analyte.encode('utf-8'))[0]
            df_temp1 = pd.DataFrame(df_temp.as_matrix()[idx],columns=df_temp.columns)

            # scatter plot before data cleaning:
            x = np.arange(len(df_temp1['FINAL'].tolist()))
            plt.figure()
            plt.scatter(x,df_temp1['FINAL'].tolist(),color='b')
            title = str(df_mat_desc['material_desc'][ind].tolist()[0]) + '||' + str(analyte)
            plt.title('Scatter plot')
            plt.xlabel('Index time')
            plt.ylabel('Values')
            plt.legend()
            plt.savefig('../infoMatcode/' + folder + '/' + str(anal) + '/brut/' + 'scatterplot.jpg')

            # clean data:
            Low_tuck,Upp_tuck,mean,std = utils.summary(df_temp1['FINAL'].tolist())
            df_out_sup = df_temp1[df_temp1.FINAL > (mean + 5*std)]
            df_out_inf = df_temp1[df_temp1.FINAL < (mean - 5*std)]
            df_out = pd.concat([df_out_sup,df_out_inf])
            df_concat = pd.concat([df_temp1,df_out])
            df_temp1 = df_concat.drop_duplicates(keep = False)       

            # scatter plot cleaned data:
            x = np.arange(len(df_temp1['FINAL'].tolist()))
            plt.figure()
            plt.scatter(x,df_temp1['FINAL'].tolist(),color='darkblue')
            title = str(df_mat_desc['material_desc'][ind].tolist()[0]) + '||' + str(analyte)
            plt.title('Scatter plot')
            plt.xlabel('Index time')
            plt.ylabel('Values')
            plt.legend()
            plt.savefig('../infoMatcode/' + folder + '/' + str(anal) + '/clean/' + 'scatterplot.jpg')