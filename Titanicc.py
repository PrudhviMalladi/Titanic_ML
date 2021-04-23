#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[63]:


train= pd.read_csv('C:/Users/prudhvi malladi/Desktop/train.csv')
test= pd.read_csv('C:/Users/prudhvi malladi/Desktop/test.csv')


# In[64]:


train.head()


# In[65]:


train.isnull()


# In[66]:


sns.heatmap(train.isnull(),yticklabels=False, cbar= False, cmap='viridis')


# In[67]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[68]:


sns.countplot(x='Survived',hue='Pclass',data= train, palette= 'RdBu_r')


# In[69]:


sns.countplot(x='Survived',hue='Sex',data= train, palette= 'rainbow')


# In[70]:


sns.distplot(train['Age'].dropna(),kde= False, color='darkred',bins=40)


# In[71]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[72]:


def impute_age(cols):
    Age =cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
    
        elif Pclass == 2:
            return 29
    
        else:
            return 24
    
    else:
            return Age
     


# In[73]:


train['Age']= train[['Age','Pclass']].apply(impute_age,axis=1)


# In[74]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')


# In[75]:


train.drop('Cabin',axis=1,inplace= True)


# In[76]:


train.head()


# In[77]:


train.info()


# In[78]:


pd.get_dummies(train['Embarked'],drop_first= True).head()


# In[79]:


Sex=pd.get_dummies(train['Sex'],drop_first= True)
Embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[80]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=True, inplace=True)


# In[81]:


train.head()


# In[82]:


train=pd.concat([train,Sex,Embark],axis=1)


# In[83]:


train.head()


# In[91]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# In[92]:


train.drop('Survived',axis=1).head()


# In[93]:


train['Survived'].head()


# In[94]:


from sklearn.model_selection import train_test_split


# In[96]:


x = train.drop(["Survived"] , axis =1)
y = train.Survived
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state = 100 , test_size = 0.3)


# In[99]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix


# In[102]:


logmodel=  LogisticRegression()
logmodel.fit(x_train,y_train)


# In[103]:


predictions = logmodel.predict(x_test)


# In[ ]:




    


# In[104]:


accuracy = confusion_matrix(y_test,predictions)


# In[105]:


accuracy


# In[106]:


from sklearn.metrics import accuracy_score


# In[108]:


accuracy=accuracy_score(y_test, predictions)
accuracy


# In[109]:


predictions


# In[ ]:




