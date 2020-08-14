#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../data/avocado.csv',index_col=0)


# In[2]:


data


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data["region"].unique()


# In[7]:


sns.barplot(x=data["year"],y=data["AveragePrice"])


# In[8]:


sns.boxplot(y="type", x="AveragePrice", data=data)


# In[9]:


a=data[data["region"]=='NewYork']
sns.boxplot(y=a["region"], x=a["AveragePrice"], data=data)


# In[10]:


Type=data.groupby('type')['Total Volume'].agg('sum')

values=[Type['conventional'],Type['organic']]
labels=['conventional','organic']
plt.figure(figsize=(14,10))
plt.pie(values,labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[11]:


Year = data['Total Volume'].groupby(data.year).sum()
Year.plot(kind='line', fontsize = 14,figsize=(14,8))
plt.show()


# In[12]:


plt.figure(figsize=(14,10))
sns.scatterplot(x='Total Volume', y = 'AveragePrice', hue= 'type', data= data)


# In[13]:


region_list=list(data.region.unique())
average_price=[]

for i in region_list:
    x=data[data.region==i]
    region_average=sum(x.AveragePrice)/len(x)
    average_price.append(region_average)

df1=pd.DataFrame({'region_list':region_list,'average_price':average_price})
new_index=df1.average_price.sort_values(ascending=False).index.values
sorted_data=df1.reindex(new_index)

plt.figure(figsize=(24,10))
ax=sns.barplot(x=sorted_data.region_list,y=sorted_data.average_price)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price of Avocado According to Region')


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

x = data.drop(['type','region','Date'], axis = 1)
y = data.type

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)


# In[15]:


logreg =  LogisticRegression(penalty='l2', tol=0.0001).fit(x_train,y_train)
print("LogisticRegression train data score:{:.3f}".
     format(logreg.score(x_train,y_train)))
print("LogisticRegression test data score:{:.3f}".
     format(logreg.score(x_test,y_test)))


# In[16]:


rf = RandomForestClassifier(n_jobs=-1,n_estimators=50,max_depth=90)
rfmodel=rf.fit(x_train,y_train)
y_pred = rfmodel.predict(x_test)
print('Accuracy: ',round((y_pred==y_test).sum()/len(y_test),3))

