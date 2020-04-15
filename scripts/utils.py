import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def get_genealogy(PATH_GEN):

    # column names setting:
    names = []
    j = 0
    for i in range(40):
        if i%4==0:
            names.append('batch_code.'+ str(j))
        if i%4 == 1:
            names.append('material_code.'+ str(j))
        if i%4 == 2:
            names.append('material_desc.'+ str(j))
        if i%4 == 3:
            names.append('ordre_fabrication.' + str(j))
            j+=1

    # load the genealogy table of the valve:
    df_gen = pd.read_csv(PATH_GEN,delimiter=';',encoding='utf-8',names=names)

    return(df_gen)

def get_matcode_level(PATH_GEN):

    # load the genealogy table of the valve:
    df_gen = get_genealogy(PATH_GEN)
    
    list_materials = []
    list_levels = []
    list_all_mat = []

    # browse all columns: 
    for col in df_gen:

        # change type of each column to string:
        df_gen[col] = df_gen[col].astype(str)

        if "material_code" in col:
            material_code = np.unique(df_gen[col]).tolist()
            if "nan" in material_code:
                material_code.remove("nan")
                
            material_code = list(map(float, material_code))
            material_code = list(map(int, material_code))
            material_code = list(map(str, material_code))
            
            list_all_mat = list_all_mat + material_code
            
            list_levels.append(col[-1])
            list_materials.append(material_code)
        
    list_all_mat = np.unique(list_all_mat).tolist() 
    df_levels = pd.DataFrame()
    df_levels['level'] = list_levels
    df_levels['material_code'] = list_materials
    df_levels.to_csv('../data/matcode_level.csv',sep=';')
    return(df_levels, list_all_mat)

def get_matcode_desc(PATH_GEN):

    # load the genealogy table of the valve:
    df_gen = get_genealogy(PATH_GEN)
    frames = []
    for i in range(10):
        mean_df = df_gen[['material_code.'+str(i),'material_desc.'+str(i)]]
        mean_df['material_code.'+str(i)] = df_gen['material_code.'+str(i)]

        mean_df.columns=['material_code','material_desc']
        frames.append(mean_df)

    df_concat = pd.concat(frames)

    df_desc = pd.DataFrame()
    df_desc['material_code'] = df_concat['material_code'].tolist()
    df_desc['material_desc'] = df_concat['material_desc'].tolist()

    df_desc = df_desc.drop(np.where(df_desc['material_code']=='nan')[0].tolist())

    df_desc = df_desc.drop_duplicates()

    df_mat_desc = pd.DataFrame()
    df_mat_desc['material_code'] = df_desc['material_code'].tolist()
    df_mat_desc['material_desc'] = df_desc['material_desc'].tolist()

    df_mat_desc.to_csv('../data/matcode_desc.csv',encoding='utf-8',sep=';')
    return(df_mat_desc)

def get_batch_desc(PATH_GEN):
    
    # load the genealogy table of the valve:
    df_gen = get_genealogy(PATH_GEN)
    frames = []
    for i in range(10):
        mean_df = df_gen[['batch_code.'+str(i),'material_desc.'+str(i)]]
        mean_df['batch_code.'+str(i)] = df_gen['batch_code.'+str(i)]

        mean_df.columns=['batch_code','material_desc']
        frames.append(mean_df)

    df_concat = pd.concat(frames)

    df_desc = pd.DataFrame()
    df_desc['batch_code'] = df_concat['batch_code'].tolist()
    df_desc['material_desc'] = df_concat['material_desc'].tolist()

    df_desc = df_desc.drop(np.where(df_desc['batch_code']=='nan')[0].tolist())

    df_desc['batch_code'] = df_desc['batch_code'].astype(str)
    df_desc = df_desc.drop_duplicates()

    df_bat_desc = pd.DataFrame()
    df_bat_desc['batch_code'] = df_desc['batch_code'].tolist()
    df_bat_desc['material_desc'] = df_desc['material_desc'].tolist()

    df_bat_desc.to_csv('../data/batch_desc.csv',encoding='utf-8',sep=';')
    return(df_bat_desc)

def get_res(PATH_RES, PATH_GEN):

    names = ['MATCODE','DATE','OPERATION_CD','TEST_CD','TEST_DS','ANALYTE','REPETITION_CD','FINAL','BATCH','SOURCE','SITE','LOW_SPEC','HIGH_SPEC','BATCHSUP','ORDNO','PRUEFLOS','STATUT_LOT','STATUT_RES']
    df_res = pd.read_csv(PATH_RES,delimiter=';',encoding='utf-8',names=names)

    if str(df_res["MATCODE"].tolist()[0]) == "MATCODE":
        df_res.index = np.arange(len(df_res))
        df_res = df_res.drop(df_res.index[0])

    df_res['FINAL'] = df_res['FINAL'].astype(str)
    df_res = df_res.drop(np.where(df_res['FINAL'] == 'nan')[0]).reset_index(drop=True)
    df_res = df_res.drop(np.where(df_res['FINAL'] == '<L.O.Q')[0]).reset_index(drop=True)
    df_res['FINAL'] = df_res['FINAL'].astype(float)
    df = df_res.groupby(['BATCH','ANALYTE'], as_index=False)['FINAL'].mean()
    df['ANALYTE'] = df['ANALYTE'].astype(str)
    df['BATCH'] = df['BATCH'].astype(str)
    df_mean_res = get_batch_desc(PATH_GEN)
    df_mean_res['batch_code'] = df_mean_res['batch_code'].astype(str)
    df_mean_res['material_desc'] = df_mean_res['material_desc'].astype(str)
    df_mean_res = df_mean_res.join(df.set_index('BATCH'), on='batch_code').reset_index().drop(columns = 'index')
    df_mean_res = df_mean_res.drop(np.where(df_mean_res['ANALYTE'] == 'nan')[0]).reset_index().drop(columns = 'index')
    df_mean_res = df_mean_res.drop(np.where(df_mean_res['material_desc'] == 'nan')[0]).reset_index().drop(columns = 'index')
    df_mean_res.to_csv('../data/mean_res.csv',encoding='utf-8',sep=';')

    return(df_res, df_mean_res)

def get_dataset(PATH_DATASET):
    df_dataset = pd.read_csv(PATH_DATASET,delimiter='\t', encoding='utf-8').drop(columns='Batch')

    names = ['X' + str(x) for x in range(len(df_dataset.columns))]

    df_column_name = pd.DataFrame()
    df_column_name['columns'] = names
    df_column_name['names'] = df_dataset.columns  

    df_column_name.to_csv('../data/columns_name.csv',sep=';', encoding='utf-8')

    df_dataset.columns = names

    return(df_dataset,df_column_name)

def quartiles(dataPoints):
    if not dataPoints:
        raise StatsError('no data points passed')
    sortedPoints = sorted(dataPoints)
    mid = len(sortedPoints) // 2 

    if (len(sortedPoints) % 2 == 0):
        lowerQ = np.median(sortedPoints[:mid])
        upperQ = np.median(sortedPoints[mid:])
    else:
        lowerQ = np.median(sortedPoints[:mid])  
        upperQ = np.median(sortedPoints[mid+1:])

    return (lowerQ, upperQ)

def summary(dataPoints):
    if not dataPoints:
        raise StatsError('no data points passed')

    Q1 = quartiles(dataPoints)[0]
    Q3 = quartiles(dataPoints)[1]
    
    Upp_tuck = Q3 + 1.5*(Q3 - Q1)
    Low_tuck = Q1 - 1.5*(Q3 - Q1)
    
    return(round(Low_tuck,3),round(Upp_tuck,3),round(np.mean(dataPoints),3),round(np.std(dataPoints),3))
