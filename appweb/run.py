import os
import sys
import numpy as np
import argparse
import argparse
import imutils
from flask import request,Flask,render_template,jsonify,Response, make_response
from flask_bootstrap import Bootstrap
import shutil
from sklearn.metrics import classification_report
from sklearn import ensemble, utils, linear_model, tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn
import pandas as pd
from sklearn.utils import shuffle
import base64
from importlib import reload
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
print(sklearn.__version__)

reload(sys)
  
PATH_DATASET = '../data/datasetReady.csv'
df_results = pd.read_csv(PATH_DATASET, encoding='utf-8', delimiter=';').drop(columns='Unnamed: 0')
df_test_dataset =  pd.read_csv('../data/datasetNotUsed.csv', encoding='utf-8', delimiter=';').drop(columns='Unnamed: 0')

#%% Run web site:
app=Flask(__name__)

Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/topIcqa',methods=['POST'])
def topIcqa():
    return render_template('topIcqa.html')

@app.route('/startPrediction',methods=['POST'])
def startPrediction():
    return render_template('startPrediction.html')

@app.route('/getInformations',methods=['POST'])
def getInformations():
    PATH = '../infoMatcode/'
    list_matcode = os.listdir(PATH)
    return render_template('getInformations.html',list_matcode=sorted(list_matcode))

@app.route('/getDistribution',methods=['POST'])
def getDistribution():

    PATH = '../infoMatcode/'
    selectMatcode = request.form.get('selectMatcode')
    print((selectMatcode))

    if str(selectMatcode) != 'None':
        PATH = '../infoMatcode/' + str(selectMatcode) + '/'
    else:
        PATH = '../infoMatcode/10055073/'
    
    list_analyte = os.listdir(PATH)
    new_list =[]
    for analyte in list_analyte:
        new_list.append(str(analyte))
    
    print(sorted(new_list))

    return render_template('getDistribution.html',list_analyte=sorted(new_list),selectMatcode=selectMatcode)

@app.route('/plot',methods=['POST','GET'])
def plot():

    print('------------------plot----------------')
    selectMatcode = request.form.get('selectedMatcode')
    print((selectMatcode))

    PATH = 'static/img/infoMatcode/' + str(selectMatcode) + '/'
    
    selectAnalyte = request.form.get('selectAnalyte')
    print((selectAnalyte))

    if '%' in str(selectAnalyte):
        selectAnalyte = selectAnalyte.replace('%','')
        
    PATH_dist = PATH + str(selectAnalyte) + '/brut/distribution.jpg'
    PATH_scatter = PATH + str(selectAnalyte) + '/brut/scatterplot.jpg'
    PATH_info = PATH + str(selectAnalyte) + '/statistics.txt'

    with open(PATH_info, "r") as f:
        content = f.read()

    title = str(selectMatcode) + ' || ' + str(selectAnalyte)
    print( content)
    print(PATH_dist)

    resp = make_response(render_template('plot.html',distribution=PATH_dist, scatterplot=PATH_scatter, statistics=content, title=title) )
    resp.cache_control.no_cache = True
    
    return resp

@app.route('/prediction',methods=['POST'])
def prediction():
        
    print('----------------------------------------------------------------')
    # get the value of the new data:
    new_data = []
    for i in range(10):
        test = 'Xtest' + str(i+1)
        value = float(request.form.get(test))
        new_data.append(value)

    print('\n\nData entred : ',new_data)

    # get dataset:
    X = df_results.drop(columns=['Y'])
    y = np.asarray(list(map(int,df_results['Y'].tolist())))
    print( '\nlength dataset  :  ',len(y))
    print('head of dataset  :  \n',X.head())


    # add new data to dataset:
    mat = np.zeros((X.shape[0]+1,X.shape[1]))
    mat[:X.shape[0],:] = X
    mat[-1] = new_data
    print('\n*** Add new data to dataset ***')
    print('new length dataset : ',len(mat))

    # scale data:
    scaler = StandardScaler()
    Xscaler = scaler.fit_transform(mat)
    print('new data scaled  : ', Xscaler[0])

    gamma = 1.3
    clf = svm.SVC(probability=True, gamma=gamma, kernel='rbf',C=10000)

    # cross validation: 
    Xvalid = Xscaler[:X.shape[0]]
    scores = cross_val_score(clf, Xvalid, y, cv=5)
    print( '\ncross validation scores :  ',scores)
    print( 'mean cross val scores   :  ',round(np.mean(scores),2))

    # split data into train and test datasets:
    x_train, x_test, y_train, y_test = train_test_split(Xvalid, y, test_size=0.01)

    # train the model:
    clf.fit(x_train ,y_train)

    #predict values:
    pred = clf.predict(x_test)
    pred_proba = clf.predict_proba(x_test)
    score = clf.score(x_test, y_test)
    print('\nsplit score : ', score)

    # predict label of new data:
    x_test[-1] = Xscaler[-1]
    pred = clf.predict(x_test)
    pred_proba = clf.predict_proba(x_test)
    
    print('\nprobas   :  ',pred_proba[-1],'\n')

    if pred[-1] == 0:
        path_image = '/static/img/5_5.png'
    else:
        path_image = '/static/img/1_5.png'

    return render_template('startPrediction.html', prediction=pred[-1], showResult = path_image)


if __name__=='__main__':
    app.run(debug=True,port='8000')
     

