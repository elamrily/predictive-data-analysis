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
            df_temp4 = pd.DataFrame(df_temp1)

            # distribution plot before data cleaning:
            kde=False
            bins=30
            counts, edges, plot = plt.hist(df_temp1['FINAL'].tolist(), bins=30)
            plt.figure()
            sns.set()
            sns.distplot(df_temp1['FINAL'].tolist(),color='b',kde=kde,bins=bins,hist_kws={'edgecolor':'black'});
            title = str(df_mat_desc['material_desc'][ind].tolist()[0]) + '||' + str(analyte)
            plt.title('Histogram')
            plt.xlabel('Test values')
            plt.ylabel('Count')
            if str(df_temp1['LOW_SPEC'][0]) != 'nan':
                plt.vlines(x=df_temp1['LOW_SPEC'][0], ymin=0, ymax=max(counts), color='red',label='Lower specification')
            
            if str(df_temp1['HIGH_SPEC'][0]) != 'nan':
                plt.vlines(x=df_temp1['HIGH_SPEC'][0], ymin=0, ymax=max(counts), color='red',label='Higher specification')
            plt.grid()
            plt.legend()
            plt.savefig('../infoMatcode/' + folder + '/' + str(anal) + '/brut/' + 'distribution.jpg')

            # Necessary statistics and data cleaning:
            nb_points = int(len(df_temp1['FINAL']))
            nb_batchs = int(len(np.unique(df_temp1['BATCH'])))
            low_spec = df_temp1['LOW_SPEC'][0]
            high_spec = df_temp1['HIGH_SPEC'][0]
            min_val = df_temp1['FINAL'].min()
            max_val = df_temp1['FINAL'].max()
            Low_tuck,Upp_tuck,mean,std = utils.summary(df_temp1['FINAL'].tolist())
            df_out_sup = df_temp1[df_temp1.FINAL > (mean + 5*std)]
            df_out_inf = df_temp1[df_temp1.FINAL < (mean - 5*std)]
            df_out = pd.concat([df_out_sup,df_out_inf]) 
            nb_outliers = float(len(df_out))
            rate = round(float((nb_outliers/nb_points))*100,3)
            df_concat = pd.concat([df_temp1,df_out])
            df_temp1 = df_concat.drop_duplicates(keep = False)
            new_mean = round(np.mean(df_temp1['FINAL'].tolist()),3)
            nb_new_data = float(len(df_temp1))
            
            # Write statistics in a file:
            file1 = open('../infoMatcode/' + folder + '/' + str(anal) + '/' + 'statistics.txt',"w") 
            L = ["Number of material tested          : "+str(nb_points)+"\n",
                "Number of batch tested             : "+str(nb_batchs)+"\n\n",
                "Minimum value of tests             : "+str(min_val)+"\n",
                "Maximum value of tests             : "+str(max_val)+"\n",
                "Mean value of tests                : "+str(mean)+"\n",
                "Standard deviation                 : "+str(std)+"\n\n",
                "Lower specification                : "+str(low_spec)+"\n",
                "Higher specification               : "+str(high_spec)+"\n",
                "Estimated lower specification      : "+str(Low_tuck)+"\n",
                "Estimated higher specification     : "+str(Upp_tuck)+"\n\n",
                "Lower limit (mean-5*std)           : "+str(mean-5*std)+"\n",
                "Higher limit (mean+5*std)          : "+str(mean+5*std)+"\n",
                "Number of outliers (mean +- 5std)  : "+str(nb_outliers)+"\n",
                "Rate of outliers (mean +- 5std)    : "+str(rate)+" % \n",
                "Mean value without outliers        : "+str(new_mean)+"\n"] 

            file1.write("\n") 
            file1.writelines(L) 
            file1.close()

            # distribution plot after data cleaning:
            counts, edges, plot = plt.hist(df_temp1['FINAL'].tolist(), bins=30)
            plt.figure()
            sns.distplot(df_temp1['FINAL'].tolist(),color='b',kde=kde,bins=bins,hist_kws={'edgecolor':'black'});
            title = str(df_mat_desc['material_desc'][ind].tolist()[0]) + '||' + str(analyte)
            plt.title('Histogram')
            plt.xlabel('Test values')
            plt.ylabel('Count')
            if str(df_temp1['LOW_SPEC'][0]) != 'nan':
                plt.vlines(x=df_temp1['LOW_SPEC'][0], ymin=0, ymax=max(counts), color='red',label='Lower specification')
            
            if str(df_temp1['HIGH_SPEC'][0]) != 'nan':
                plt.vlines(x=df_temp1['HIGH_SPEC'][0], ymin=0, ymax=max(counts), color='red',label='Higher specification')
            plt.grid()
            plt.legend()
            plt.savefig('../infoMatcode/' + folder + '/' + str(anal) + '/clean/' + 'distribution.jpg')