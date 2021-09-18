#!/usr/bin/python3

import cgi

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import mysql.connector
import getpass

form =  cgi.FieldStorage()
print("content-type: text/html")

print()

disease = pd.read_csv('Training.csv') 
df = disease.copy()
sql_Copy = disease.copy()
df.rename(columns={"fluid_overload.1":"fluid_overload1"}, inplace=True)
df.drop(['Unnamed: 133'],axis=1,inplace=True)

sql_Copy.drop('prognosis', axis=1,inplace=True)
sql_Copy.drop('Unnamed: 133', axis=1,inplace=True)
sql_Copy.rename(columns={"fluid_overload.1":"fluid_overload1"}, inplace=True)

userd = form.getvalue("username")
Password = form.getvalue("Password")
conn = mysql.connector.connect(user=userd,password=Password,host="localhost")
cur = conn.cursor(prepared=True)
cur.execute("create database if not exists test")
conn = mysql.connector.connect(user=userd,password=Password,host="localhost",database="test")
cur = conn.cursor(prepared=True)

col = [i.replace(' ','') for i in sql_Copy.columns]
sql_Copy.columns = col
col = [i.replace('(','') for i in sql_Copy.columns]
sql_Copy.columns = col
col = [i.replace(')','') for i in sql_Copy.columns]
sql_Copy.columns = col


name = form.getvalue("name")
phone = form.getvalue("phone")
address = form.getvalue("address")
lst = form.getvalue("list")
first_time = form.getvalue("i")

if first_time == "first time":
    cur.execute('''drop table if exists disease''')
    cur.execute('''create table if not exists disease (phone_no BIGINT(12) primary key,name VARCHAR(255), address VARCHAR(255))''')
    for i in sql_Copy.columns:
        cur.execute('alter table disease add column {} INT DEFAULT 0'.format(i))

else:
   print("data aval")

cur.execute('''insert into disease(phone_no,name,address) values ({},'{}','{}')'''.format(phone,name,address))
conn.commit()

for i in lst:
    cur.execute("update disease set {} = {} where phone_no = {}".format(i,1,phone))
    conn.commit()


cur.execute('''select * from disease where phone_no = {}'''.format(phone))
var2 = cur.fetchone()

prid = []
for i in var2:
    prid.append(i)


prid_data = np.array(prid[3:])
prid_data = prid_data.reshape(-1,132)


#model training
X_train = df.drop('prognosis', axis=1)
X_test = df.drop('prognosis', axis=1)
y_train = np.array(df['prognosis'])

Xnew= df.drop('prognosis', axis=1)
ynew= df['prognosis']

y_train_enc = pd.get_dummies(y_train)

def create_model(learning_rate, activation):
    model2= Sequential()
    my_opt= Adam(lr= learning_rate)
    model2.add(Dense(64, activation=activation, input_shape=(X_train.shape[1],)))
    model2.add(Dense(y_train_enc.shape[1], activation='softmax'))
    model2.compile(optimizer= my_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model2

modelnew = KerasClassifier(build_fn= create_model, epochs=30, batch_size=100, validation_split=0.3)

params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [10], 'learning_rate': [0.1, 0.01, 0.001]}
random_search = RandomizedSearchCV(modelnew, param_distributions = params, cv = 5)
random_search.fit(Xnew, ynew, verbose=0)

random_search.best_estimator_.fit(Xnew, ynew,verbose=0)

prediction= random_search.best_estimator_.predict(prid_data)
print("<br><h1> {} <h1>".format(prediction[0]))
