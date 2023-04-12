import pymongo
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    connection = MongoClient('localhost', 27017)
    db = connection.HeartAttack

    data = db.HeartAttack
    Heartattacklist=data.find()

    cursor = data.find()
    entries = list(cursor)

    df = pd.DataFrame(entries)

    df = df.loc[:, df.columns != '_id']

    X = df.drop("target", axis = 1)
    Y = df['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

   

    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    value1 = float(request.GET['n1'])
    value2 = float(request.GET['n2'])
    value3 = float(request.GET['n3'])
    value4 = float(request.GET['n4'])
    value5 = float(request.GET['n5'])
    value6 = float(request.GET['n6'])
    value7 = float(request.GET['n7'])
    value8 = float(request.GET['n8'])
    value9 = float(request.GET['n9'])
    value10 = float(request.GET['n10'])
    value11 = float(request.GET['n11'])
    value12 = float(request.GET['n12'])
    value13 = float(request.GET['n13'])



    pred = model.predict([[value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, value11, value12, value13]])

    result1 = ""
    if pred==[1]:
        result1 = "Heartattack possible"
    else:
        result1 = "Not prone to heartattack"

    return render(request, "predict.html", {"result2": result1})