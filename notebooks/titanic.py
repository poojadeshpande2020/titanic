#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import pickle

np.set_printoptions(suppress = True)

plt.rcParams['figure.figsize'] = 8,5


# In[4]:


train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


# In[5]:


df = train_df


# In[6]:


df.head()


# ## **Basic EDA and Visualizations**

# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


#Check for null values
sns.heatmap(df.isnull(),yticklabels='None')
plt.show()


# In[11]:


#Check how many travellers survived and how many didn't
sns.countplot('Survived',data = df,palette='inferno')
plt.show()


# 
# 
# 
# 
# Imbalanced dataset. Positive class is lower than the negative

# In[12]:


#Analysis of Sex and Survived
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
sns.countplot('Sex',data=df,ax=ax1,palette='RdBu')
ax1.set_title("Ratio of males to females on titanic")
ax2 = fig.add_subplot(1,2,2)
sns.barplot('Sex','Survived',data = df,ax=ax2,palette='icefire')

ax2.set_title("Number of male and female survivors")

plt.show()


# More males in the dataset but more number of female survivors. Let's see the male and female survival rate

# In[13]:


#Calculating the male and female survival rate
total_male = (train_df['Sex']=='male').sum()
total_male_survived = ((train_df['Sex']=='male')&(train_df['Survived']==1)).sum()

total_female = (train_df['Sex']=='female').sum()
total_female_survived = ((train_df['Sex']=='female')&(train_df['Survived']==1)).sum()

print("Male survival rate : {:.2f}%".format(total_male_survived*100/total_male))
print("Female survival rate : {:.2f}%".format(total_female_survived*100/total_female))


# In[14]:


#Amongst the people that survived, how many male and female
survived = train_df[train_df['Survived']==1][['Survived','Sex']]

sns.countplot('Sex',data=survived,palette='muted')
plt.show()


# In[15]:


#Age of the people that survived and that didn't
sns.boxplot('Survived','Age',data=df,palette='Set3')
plt.show()


# Most of the people between the ages of 20 and 40

# In[16]:


#Analyzing the distribution of fare amongst survivors and non survivors
survived_fare = train_df[train_df['Survived']==1]['Fare']
notsurvived_fare = train_df[train_df['Survived']==0]['Fare']


# In[17]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
sns.distplot(survived_fare,kde=False,hist_kws=dict(edgecolor="k", linewidth=2),bins =20,ax=ax1,color='darkblue')
ax1.set_title("Survived = 1")
ax2 = fig.add_subplot(1,2,2)
sns.distplot(notsurvived_fare,kde=False,hist_kws=dict(edgecolor="k", linewidth=2),bins =20,ax=ax2,color='darkred')
ax2.set_title("Survived = 0")
ax2.set_xlim(0,500)
plt.show()


# People who paid higher fare have survived than the ones who paid lower fare or you could say richer people survived.

# In[18]:


#Analyzing Pclass
sns.barplot('Sex','Survived',hue='Pclass',data=train_df,palette='Set1')
plt.show()


# As expected, 1st class travellers have survived more than second/third class survivors

# In[19]:


#Where did the survivors embark
sns.barplot('Sex','Survived',hue='Embarked',data=train_df,palette='Set2')
plt.show()


# People who embarked on Cherbourg have higher survival percentage

# In[20]:


sns.barplot('Embarked','Survived',hue = 'Pclass',data = train_df,palette='Set1')


# In[21]:


sns.catplot(x="Fare", y="Survived", row="Pclass",
                kind="box", orient="h", height=1.5, aspect=4,
                data=train_df.query("Fare > 0"),palette='husl')
plt.show()


# In[22]:


sns.scatterplot('Age','Fare',hue = 'Survived',data=train_df,palette = 'gist_earth')
plt.title("Age vs Fare")
plt.show()


# A few outliers in Fare. These could be influential for the problem. May consider removing those while building the models

# In[23]:


sns.distplot(train_df[train_df['Age'].notnull()]['Age'],bins=20,kde=True,hist_kws=dict(edgecolor="k", linewidth=2),color='DarkBlue')
plt.title("Histogram of Age")
plt.show()


# In[24]:


sns.pointplot('Pclass','Age',hue = 'Survived',data = train_df)
plt.title("Mean age of the survivors and non survivors per Class")
plt.show()


# ### **Creating new features**
# 

# In[25]:


test_df.isna().sum()


# In[26]:


#Create family size as a linear combination of sibling, spouse, parents and children
df['family_size'] = df['SibSp']+df['Parch']


# In[27]:


#Relationship of family size and survived
sns.barplot('family_size','Survived',data = train_df)
plt.title("Family size and survived")
plt.show()


# Family size of 3 have had a higher survival rate than family sizes of zero. Family size of 5 had the lowest survival percentage. Family sizes of over 6 have not survived.

# In[28]:


#Create title feature
def find_substrings(main_string,substrings):
  for substring in substrings:
    if (main_string.find(substring))!=-1:
      return substring
  return np.nan


# In[29]:


titles_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']


# In[30]:


df['title'] = df['Name'].apply(lambda x: find_substrings(x,titles_list))


# In[31]:


def replace_title(x):
  title = x['title']
  if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
      return 'Mr'
  elif title in ['Countess', 'Mme']:
      return 'Mrs'
  elif title in ['Mlle', 'Ms']:
      return 'Miss'
  elif title =='Dr':
      if x['Sex']=='Male':
          return 'Mr'
      else:
          return 'Mrs'
  else:
      return title


# In[32]:


df['title'] = df.apply(replace_title,axis = 1)


# # Dropping rows with extreme outliers in 'Fare' column. After doing this, noticed an increase in cross validation scores but the test case scores for the three selected models have decreased 

# In[33]:


# indices = df['Fare'][df['Fare']>400].index
# df.drop(indices,inplace = True)


# In[34]:


#Drop 'Name','PassengerId', 'Ticket'
df.drop(columns = ['Name','PassengerId','Ticket','Cabin'],inplace = True)


# In[35]:


#Encoding 'Sex' column
sex_dict = {'male':0,'female':1}
df['Sex'] = df['Sex'].map(sex_dict)


# In[36]:


#Encoding 'Embark' column
embark_dict = {'C':0,'Q':1,'S':2}
df['Embarked'] = df['Embarked'].map(embark_dict)


# In[37]:


#Encoding title column
title_dict = {"Mr":0,"Mrs":1,"Miss":2,"Master":3}
df['title'] = df['title'].map(title_dict)


# In[38]:


#Remove Siblings, spouses, parent and children columns
#train_df.drop(columns = ['SibSp','Parch'],inplace = True)


# In[39]:


df.head()


# # Age fill na with mean age based on the title

# In[40]:


mean_age_0 = df['Age'][df['title']==0].mean()
mean_age_1 = df['Age'][df['title']==1].mean()
mean_age_2 = df['Age'][df['title']==2].mean()
mean_age_3 = df['Age'][df['title']==3].mean()


# In[41]:


df.loc[df['title']==0,'Age'] = df.loc[df['title']==0,'Age'].fillna(mean_age_0)
df.loc[df['title']==1,'Age'] = df.loc[df['title']==1,'Age'].fillna(mean_age_1)
df.loc[df['title']==2,'Age'] = df.loc[df['title']==2,'Age'].fillna(mean_age_2)
df.loc[df['title']==3,'Age'] = df.loc[df['title']==3,'Age'].fillna(mean_age_3)


# In[42]:



df[df['Embarked'].isna()]


# In[43]:


df.isna().sum()


# In[44]:



#instantiate both packages to use

encoder = OrdinalEncoder()
imputer = KNN()
# create a list of categorical columns to iterate over
cat_cols = ['Embarked','Pclass','Sex','Survived','title']

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode data
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(df[columns])


# In[45]:


encode_data = pd.DataFrame(np.round(imputer.fit_transform(df)),columns = df.columns)


# In[46]:


for col_name in cat_cols:  
    # Perform inverse transform of the ordinally encoded columns
    encode_data[col_name] = encoder.inverse_transform(encode_data[col_name].values.reshape(-1,1))


# In[47]:


encode_data.head()


# In[48]:


X = encode_data.iloc[:,1:].values
y = encode_data.iloc[:,0].values


# In[49]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 123)


# # K nearest neighbors

# In[50]:


knn = KNeighborsClassifier()


# In[51]:


knn.fit(X_train,y_train)


# In[52]:


scores = cross_validate(knn,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[53]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# # Decision Tree

# In[54]:



dtree = DecisionTreeClassifier()


# In[55]:


dtree.fit(X_train,y_train)


# In[56]:


scores = cross_validate(dtree,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[57]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# # Support Vector Machine

# In[58]:


svm = SVC()


# In[59]:


svm.fit(X_train,y_train)


# In[60]:


scores = cross_validate(svm,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[61]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# # Based on cross validation scores, the following models were selected

# # Logistic Regression Model

# In[62]:


lr = LogisticRegression(max_iter = 1000)


# In[63]:


lr.fit(X_train,y_train)


# In[64]:


scores = cross_validate(lr,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[65]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# In[66]:


param_grid = {'C':[0.001,0.01,0.1,1,10,100]}

grid = GridSearchCV(lr,param_grid,refit=True,verbose=2,scoring = "accuracy")
grid.fit(X_train,y_train)
print(grid.best_estimator_)
print(grid.best_params_)


# In[67]:


lr = LogisticRegression(max_iter = 1000,C = 0.1)


# In[68]:


lr.fit(X_train,y_train)


# In[69]:


#Predict on training and test data

pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)


# In[70]:


#Evaluation

print("=====Accuracy score============\n")
print("Train accuracy score : {:.4f}".format(accuracy_score(y_train,pred_train)))
print("Test accuracy score : {:.4f}".format(accuracy_score(y_test,pred_test)))
print("\n=====Confusion Matrix==========\n ")
print(confusion_matrix(y_test,pred_test))
print("\n=====Classification report=====\n")
print(classification_report(y_test,pred_test))


# In[71]:


plot_confusion_matrix(lr,X_test,y_test,values_format='d',cmap = 'Blues')
plt.show()


# In[72]:


probs = lr.predict_proba(X_test)[:,1]


# In[73]:


fpr,tpr,thresholds = roc_curve(y_test,probs)

plt.plot(fpr,tpr,linewidth = 4)
plt.plot([0,1],[0,1],linewidth = 4)
plt.title("ROC curve for logistic regression")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[74]:


score = roc_auc_score(y_test,probs)
print("ROC AUC score : {}".format(round(score,4)))


# In[75]:


#Learning curve
lc = learning_curve(lr,X_train,y_train)

train_scores_mean = np.mean(lc[1],axis = 1)
test_scores_mean = np.mean(lc[2],axis = 1)

plt.plot(lc[0],train_scores_mean,marker = 'D',linewidth = 4,color = '#C70E41')
plt.plot(lc[0],test_scores_mean,marker = 'D',linewidth = 4,color = '#204FD1')
plt.legend(['Train scores','Cross validated scores'])
plt.title("Learning curve - Logistic Regression")
plt.xlabel("Training examples")
plt.ylabel("Scores")
plt.grid()
plt.show()


# # Naive Bayes model

# In[76]:


nb = GaussianNB()


# In[77]:


nb.fit(X_train,y_train)


# In[78]:


scores = cross_validate(nb,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[79]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# In[80]:


pred_train = nb.predict(X_train)
pred_test = nb.predict(X_test)


# In[81]:


print("=====Accuracy score============\n")
print("Train accuracy score : {:.4f}".format(accuracy_score(y_train,pred_train)))
print("Test accuracy score : {:.4f}".format(accuracy_score(y_test,pred_test)))
print("\n=====Confusion Matrix==========\n ")
print(confusion_matrix(y_test,pred_test))
print("\n=====Classification report=====\n")
print(classification_report(y_test,pred_test))


# In[82]:


plot_confusion_matrix(nb,X_test,y_test,values_format='d',cmap = 'Blues')
plt.show()


# In[83]:


probs = nb.predict_proba(X_test)[:,1]


# In[84]:


fpr,tpr,thresholds = roc_curve(y_test,probs)

plt.plot(fpr,tpr,linewidth = 4)
plt.plot([0,1],[0,1],linewidth = 4)
plt.title("ROC curve for naive bayes")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[85]:


score = roc_auc_score(y_test,probs)
print("ROC AUC score : {}".format(round(score,4)))


# In[86]:


#Learning curve
lc = learning_curve(nb,X_train,y_train)

train_scores_mean = np.mean(lc[1],axis = 1)
test_scores_mean = np.mean(lc[2],axis = 1)

plt.plot(lc[0],train_scores_mean,marker = 'D',linewidth = 4,color = '#C70E41')
plt.plot(lc[0],test_scores_mean,marker = 'D',linewidth = 4,color = '#204FD1')
plt.legend(['Train scores','Cross validated scores'])
plt.title("Learning curve - Naive Bayes")
plt.xlabel("Training examples")
plt.ylabel("Scores")
plt.grid()
plt.show()


# # Random Forest Classifier

# In[87]:


forest = RandomForestClassifier()


# In[88]:


forest.fit(X_train,y_train)


# In[89]:


scores = cross_validate(forest,X_train,y_train,scoring=['accuracy','f1','roc_auc'])


# In[90]:


print("Mean accuracy with cross validation : {:.2f}".format(scores['test_accuracy'].mean()))
print("Mean f1 with cross validation : {:.2f}".format(scores['test_f1'].mean()))
print("Mean ROC AUC score with cross validation : {:.2f}".format(scores['test_roc_auc'].mean()))


# In[91]:


pred_train = forest.predict(X_train)
pred_test = forest.predict(X_test)


# In[92]:


print("=====Accuracy score============\n")
print("Train accuracy score : {:.4f}".format(accuracy_score(y_train,pred_train)))
print("Test accuracy score : {:.4f}".format(accuracy_score(y_test,pred_test)))
print("\n=====Confusion Matrix==========\n ")
print(confusion_matrix(y_test,pred_test))
print("\n=====Classification report=====\n")
print(classification_report(y_test,pred_test))


# In[93]:


#Clearly overfits. So we will try to tune the hyperparameters

param_grid = {'n_estimators':[100,200,500,1000],'max_features':[2,3,4,5,6],'max_depth':[2,3,4],'min_samples_leaf':[2,3]}

grid = GridSearchCV(forest,param_grid,refit=True,verbose=2,scoring = "f1_weighted")
grid.fit(X_train,y_train)
print(grid.best_estimator_)
print(grid.best_params_)


# In[94]:


rf = RandomForestClassifier(max_depth=4,max_features=5,n_estimators=100,min_samples_leaf=3)


# In[95]:


rf.fit(X_train,y_train)


# In[96]:


pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)


# In[97]:


print("=====Accuracy score============\n")
print("Train accuracy score : {:.4f}".format(accuracy_score(y_train,pred_train)))
print("Test accuracy score : {:.4f}".format(accuracy_score(y_test,pred_test)))
print("\n=====Confusion Matrix==========\n ")
print(confusion_matrix(y_test,pred_test))
print("\n=====Classification report=====\n")
print(classification_report(y_test,pred_test))


# In[98]:


plot_confusion_matrix(rf,X_test,y_test,values_format='d',cmap = 'Blues')
plt.show()


# In[99]:


probs = rf.predict_proba(X_test)[:,1]


# In[100]:


fpr,tpr,thresholds = roc_curve(y_test,probs)

plt.plot(fpr,tpr,linewidth = 4)
plt.plot([0,1],[0,1],linewidth = 4)
plt.title("ROC curve for Random Forest")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[101]:


score = roc_auc_score(y_test,probs)
print("ROC AUC score : {}".format(round(score,4)))


# In[102]:


#Learning curve
lc = learning_curve(rf,X_train,y_train)

train_scores_mean = np.mean(lc[1],axis = 1)
test_scores_mean = np.mean(lc[2],axis = 1)

plt.plot(lc[0],train_scores_mean,marker = 'D',linewidth = 4,color = '#C70E41')
plt.plot(lc[0],test_scores_mean,marker = 'D',linewidth = 4,color = '#204FD1')
plt.legend(['Train scores','Cross validated scores'])
plt.title("Learning curve - Random Forest")
plt.xlabel("Training examples")
plt.ylabel("Scores")
plt.grid()
plt.show()


# # Pipeline for modeling

# In[103]:


passender_id = test_df['PassengerId']


# In[104]:


def create_new_features(dataframe):
  dataframe['family_size'] = dataframe['SibSp']+dataframe['Parch']
  dataframe['title'] = dataframe['Name'].apply(lambda x: find_substrings(x,titles_list))
  dataframe['title'] = dataframe.apply(replace_title,axis = 1)

  return dataframe


# In[105]:


def preprocessing(dataframe):
  global mean_age_0
  global mean_age_1
  global mean_age_2
  global mean_age_3
  dataframe.drop(columns = ['Name','PassengerId','Ticket','Cabin'],inplace = True)
  dataframe['Sex'] = dataframe['Sex'].map(sex_dict)
  dataframe['Embarked'] = dataframe['Embarked'].map(embark_dict)
  dataframe['title'] = dataframe['title'].map(title_dict)
  dataframe.loc[dataframe['title']==0,'Age'] = dataframe.loc[dataframe['title']==0,'Age'].fillna(mean_age_0)
  dataframe.loc[dataframe['title']==1,'Age'] = dataframe.loc[dataframe['title']==1,'Age'].fillna(mean_age_1)
  dataframe.loc[dataframe['title']==2,'Age'] = dataframe.loc[dataframe['title']==2,'Age'].fillna(mean_age_2)
  dataframe.loc[dataframe['title']==3,'Age'] = dataframe.loc[dataframe['title']==3,'Age'].fillna(mean_age_3)
  

  dataframe['Fare'].fillna(dataframe['Fare'].mean(),inplace = True)

  return dataframe


# In[106]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(create_new_features)
t = FunctionTransformer(preprocessing)


# In[107]:


transformer.transform(test_df)
t.transform(test_df)


# In[108]:


test_df.isna().sum()


# In[109]:


test_df['Survived'] = lr.predict(test_df.values)


# In[110]:


test_df.head()


# ## Kaggle submission file

# In[ ]:


test_df['PassengerId'] = passender_id


# In[ ]:


submission = test_df.drop(columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','family_size','title'])


# In[ ]:


submission = submission[['PassengerId','Survived']]
submission.set_index('PassengerId',drop = True,inplace = True)


# ## Pickling logistic regression model

# In[112]:


with open('model.pkl','wb') as file:
    pickle.dump(lr,file)


# In[ ]:




