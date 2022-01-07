#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
import numpy as np


# # 1. Load the data into the pandas environment and identify some basic details of the dataset.

# In[2]:


data=pd.read_csv('employee.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.isna().sum()


# # 2. Reset the index as "name" as the index. 

# In[7]:


data.set_index(['name'],inplace=True)


# In[9]:


data


# # 3. Select rows for specific names Jack Morgan and Josh wills.

# In[10]:


data.loc[['Jack Morgan','Josh Wills']]


# # 4. Select data for multiple values "Sales" and “Finance”.

# In[11]:



data.loc[(data['department']=='Sales')|(data['department']=='Finance')]


# # 5. Display employee who has more than 700 performance score

# In[12]:


data.loc[(data['performance_score']>700)]


# # 6. Display employee who has more than 500 and less than 700 performance score

# In[13]:


data.loc[(data['performance_score']>500) & (data['performance_score']<700)]


# # 7. Check and handle missing values in the dataset.

# In[14]:


data.isna().sum()


# In[15]:


data['age'].fillna(data['age'].median(),inplace=True)


# In[16]:


data['income'].fillna(data['income'].mean(),inplace=True)


# In[17]:


data['gender'].fillna(data['gender'].mode()[0],inplace=True)


# In[18]:


data.isna().sum()


# thus all null values are gone

# # 8. Check the outliers and handle outliers in performance score using Percentiles.

# In[19]:


plt.boxplot(data['performance_score'])
plt.title('Box plot of performance score')


# In[20]:


Q1=np.percentile(data['performance_score'],25,interpolation='midpoint')
Q2=np.percentile(data['performance_score'],50,interpolation='midpoint')
Q3=np.percentile(data['performance_score'],75,interpolation='midpoint')


# In[21]:


IQR=Q3-Q1


# In[22]:


low_lim=Q1-1.5*IQR
up_lim=Q3+1.5*IQR


# In[23]:


print(low_lim)
print(up_lim)


# In[24]:


outlier=[]
for i in data['performance_score']:
    if (i<low_lim) or (i>up_lim):
        outlier.append(i)


# In[25]:


outlier


# In[26]:


ind=data['performance_score']<low_lim
data.loc[ind].index


# In[27]:


data.loc[['James Authur']]


# In[33]:


data['performance_score'].hist(figsize=(10,10))


# In[34]:


data.loc[list(ind),'performance_score']=data['performance_score'].median()


# In[35]:


data


# # 9. Check the gender column and do Dummy encoding.

# In[36]:


data1=pd.get_dummies(data,columns=['gender'])


# In[37]:


data1


# # 10. Do the standard scaling on the feature performance score

# In[38]:


from sklearn import preprocessing
standardisation=preprocessing.StandardScaler()
x=standardisation.fit_transform(data[['performance_score']])
data[['performance_score']]=x
data

