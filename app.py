#!/usr/bin/env python
# coding: utf-8

# In[3]:


import flask
from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import numpy as np
import pickle



# In[ ]:

with open(f'notebooks/model.pkl','rb') as f:
    model = pickle.load(f)


app = Flask(__name__,template_folder= 'templates')

@app.route('/',methods = ['GET','POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':
        Pclass = request.form['Pclass']
        Sex = request.form['Sex']
        Age = request.form['Age']
        SibSp = request.form['SibSp']
        Parch = request.form['Parch']
        Fare = request.form['Fare']
        Embarked = request.form['Embarked']
        family_size = SibSp + Parch
        title = request.form['title']
        input_values = pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,family_size,title]],columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','family_size','title'])

        sex_dict = {'male':0,'female':1}
        input_values['Sex'] = input_values['Sex'].map(sex_dict)
        embark_dict = {'C':0,'Q':1,'S':2}
        input_values['Embarked'] = input_values['Embarked'].map(embark_dict)
        title_dict = {"Mr":0,"Mrs":1,"Miss":2,"Master":3}
        input_values['title'] = input_values['title'].map(title_dict)

        p = model.predict(input_values.values)

        if p[0] == 0:
            prediction = "No"
        else:
            prediction = 'Yes'

        return render_template('main.html',original_input = {'Pclass':Pclass,'Sex':Sex,'Age':Age,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Embarked':Embarked,'title':title},result = prediction)

if __name__ == '__main__':
    app.run()


# In[ ]:




