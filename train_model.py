# PENGIMPORTAN LIBRARY YANG DIGUNAKAN

from sklearn.svm import SVC
import numpy as np 
import pandas as pd
from sklearn import metrics

# MEMBACA DATA SET YANG DIGUNAKAN

data = pd.read_csv('dataset/DataTest.csv')

# PEMISAHAN DATA DAN  TARGET
X = data.iloc[:,0:4].values
Y = data.iloc[:,4].values

# TRAINING DATA
clf = SVC(kernel='linear')
clf.fit(X, Y)

# FUNGSI PREDIKSI WAJAH BERDASARKAN FITUR YANG TERTANGKAP OLEH KAMERA 
def predict (data_test):
    prediction = clf.predict(data_test)
    
    return prediction

    