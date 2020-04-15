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

PATH_DATASET = '../data/DATASET_FINAL.txt'

if __name__=='__main__':

    df_dataset, df_column_name = utils.get_dataset(PATH_DATASET)
    print('\n dataset head : \n',df_dataset.head())

    df_describe = df_dataset.describe()
    df_describe.to_csv('../infoDataset/dataset_description.csv', sep=';', encoding='utf-8')

    mat_dataset = df_dataset.as_matrix()
    row, col = mat_dataset.shape
    
    list_count = []
    for j in range(col):
        count = 0
        for i in range(row):
            if str(mat_dataset[i,j]) == 'nan':
                count += 1

        list_count.append(float(count))
    
    df_missing = pd.DataFrame()
    df_missing['column'] = df_dataset.columns
    df_missing['material_description|analyte'] = df_column_name['names'].tolist()
    df_missing['count missing'] = list_count
    df_missing['rate (%)'] = np.around(100 * np.asarray(list_count)/len(df_dataset), decimals=2)
    # df_missing = df_missing.sort_values(by=['count missing'],ascending=False)
    df_missing.to_csv('../infoDataset/missingValues.csv', sep=';', encoding='utf-8')
    
    # Plot the statistics
    x = np.arange(len(df_dataset.columns))
    plt.figure(figsize=(100,20))
    plt.title('Distribution des valeurs manquantes par attribut')
    plt.bar(x,list_count)
    plt.xticks(x, df_dataset.columns, rotation=90)
    plt.grid()
    plt.savefig('../infoDataset/missingValues.jpg')

    count = 1
    for i,col in enumerate(df_dataset.columns):
        if str(col) in os.listdir('../infoDataset/details/'):
            shutil.rmtree('../infoDataset/details/' + str(col))
            os.mkdir('../infoDataset/details/' + str(col))
        if str(col) not in os.listdir('../infoDataset/details/'):
            os.mkdir('../infoDataset/details/' + str(col))
    
        print('------------------------------------',col,'-------------------------------------------')
        df_dataset[col] = df_dataset[col].astype(str)
        vect_val = df_dataset[col].drop(np.where(df_dataset[col] == 'nan')[0])
        vect_val = vect_val.astype(float)

        # Necessary informations:
        kde=False
        bins=30 
        nb_points = float(len(vect_val))
        min_val = vect_val.min()
        max_val = vect_val.max()
        nb_miss = float(len(df_dataset[col].tolist()) - nb_points)
        rate = round(100*nb_miss/len(df_dataset[col].tolist()),2)

        if len(vect_val.tolist()) != 0:
            Low_tuck,Upp_tuck,mean,std = utils.summary(vect_val.tolist())
        else:
            Low_tuck = 'nan'
            Upp_tuck = 'nan'
            mean = 'nan'
            std = 'nan'
        
        print('|nb_points   : ',nb_points)
        print('|rate        : ',rate)
        
        # Write data in a file:
        file1 = open('../infoDataset/details/' +col+ '/statistics.txt',"w") 
        L = ["Number of datas    : "+str(nb_points)+"\n",
            "Min                : "+str(min_val)+"\n",
            "Max                : "+str(max_val)+"\n",
            "Mean               : "+str(mean)+"\n",
            "std                : "+str(std)+"\n",
            "Low Tukey          : "+str(Low_tuck)+"\n",
            "High Tukey         : "+str(Upp_tuck)+"\n",
            "Number of missing  : "+str(nb_miss)+"\n",
            "rate of missing (%): "+str(rate)] 

        file1.write(col + " || " + df_column_name['names'].tolist()[i]  +"\n\n") 
        file1.writelines(L) 
        file1.close()

        # Scatter plot:
        x = vect_val.index
        plt.figure()
        plt.scatter(x,vect_val.tolist(),color='darkblue')
        title = col 
        plt.title(title)
        plt.xlabel('index')
        plt.ylabel('values')
        plt.grid()
        plt.legend()
        plt.savefig('../infoDataset/details/' + col + '/scatterplot.jpg')

        # distribution plot:
        plt.figure()
        sns.distplot(vect_val,color='darkblue',kde=kde,bins=bins,hist_kws={'edgecolor':'black'});
        title = col
        plt.title(title)
        plt.xlabel('values')
        plt.ylabel('frequency')
        plt.grid()
        plt.legend()
        plt.savefig('../infoDataset/details/' + col + '/distribution.jpg')

        print('\n\t\t\t\t\t-',count,'-\n')
        count+=1
            