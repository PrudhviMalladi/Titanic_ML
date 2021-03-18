#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


train = pd.read_csv('C:/Users/prudhvi malladi/Desktop/train.csv')
test= pd.read_csv('C:/Users/prudhvi malladi/Desktop/test.csv')


# In[56]:


train.head()


# In[57]:


train1 = train.drop(['Name','Cabin','Ticket'], axis =1)


# In[58]:


train1.info()
test.info()


# In[59]:


#Distribution of null values for training data
print(train1.isnull().sum())
train1.isnull().mean()


# In[60]:


train1.isnull().sum()


# In[61]:


train1.isnull().mean()


# In[62]:


#Distribution of null values for test data
print(test.isnull().sum())
test.isnull().mean()


# In[63]:


train1.nunique()#@unique values for training data


# In[64]:


test.nunique()#@unique values for training data


# In[65]:


train1.describe()


# In[66]:


train1.corr()


# In[67]:


import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(train1.corr())# we can see it better with the heatmap


# In[68]:


sns.pairplot(train1)#let's take a look at the distributions. Many variables looks like categorical even they are numeric
plt.show()


# In[97]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV , train_test_split


# In[98]:


test.head()


# In[99]:


x = train1.drop(["Sex"] , axis =1)
y = train1.Sex
x_train1 , x_test , y_train1 , y_test = train_test_split(x,y,random_state = 100 , test_size = 0.3)


# In[100]:


train1.dropna(inplace = True)
train1.isna().sum()


# In[101]:


le = LabelEncoder()
train1["Sex"] = le.fit_transform(train1["Sex"])
train1["Embarked"] = le.fit_transform(train1["Embarked"])
train1["PassengerId"]= le.fit_transform(train1["PassengerId"])

test["Sex"] = le.fit_transform(test["Sex"])
test["Embarked"] = le.fit_transform(test["Embarked"])
test["PassengerId"]= le.fit_transform(test["PassengerId"])


# In[102]:


train1
test.head()


# In[103]:


feat = ExtraTreesRegressor()
feat.fit(x_train , y_train)


# In[104]:


features = pd.Series( feat.feature_importances_ , index = x_train.columns )
features.nlargest(10).plot(kind = "barh")
plt.show()


# In[105]:


lr = LinearRegression()
rfr = RandomForestRegressor()
dt = DecisionTreeRegressor()


# In[106]:


print(lr.fit(x_train1 , y_train1))
print(rfr.fit(x_train1 , y_train1))
print(dt.fit(x_train1 , y_train1))


# In[107]:


print(r2_score(lr.predict(x_train1) , y_train1))
print(r2_score(rfr.predict(x_train1) , y_train1))
print(r2_score(dt.predict(x_train1) , y_train1))


# In[108]:


print(r2_score(lr.predict(x_test) , y_test))
print(r2_score(rfr.predict(x_test) , y_test))
print(r2_score(dt.predict(x_test) , y_test))


# In[ ]:





# In[ ]:




