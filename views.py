from django.shortcuts import render

# Create your views here.
from django.shortcuts import render 
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from django.http import HttpResponse

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def index(request): 
	return render(request, "index.html") 


def predict(request): 
	return render(request, "predict.html") 


def result(request): 
    data = pd.read_csv('diabetes.csv')
    #X = data[['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']]
    #X = data[feature_cols] 
    X = data.drop(["Outcome"], axis=1) 
    y = data.Outcome 
    #from sklearn.model_selection import train_test_split
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    model = LogisticRegression() 
    model.fit(X, y) 
    val1 = float(request.GET['n1']) 
    val2 = float(request.GET['n2']) 
    val3 = float(request.GET['n3']) 
    val4 = float(request.GET['n4']) 
    val5 = float(request.GET['n5']) 
    val6 = float(request.GET['n6']) 
    val7 = float(request.GET['n7']) 
    val8 = float(request.GET['n8']) 
    
    pred = model.predict([[val1, val2, val3, 
						val4, val5, val6, val7,val8]]) 
    result = "" 
    if pred == [0]: 
        result = "<h1>You are not diabetic.</h1>"
    else: 
        
        result= "<h1>You are diabetic.</h1>"
    return HttpResponse(result)
