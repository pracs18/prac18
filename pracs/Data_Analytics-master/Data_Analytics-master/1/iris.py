#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
iris = pd.read_csv("Iris.csv") # the iris dataset is now a Pandas DataFrame
iris.head()
iris = iris.drop(['Id'],axis = 1)


# In[2]:


iris.info()


# In[3]:


iris.describe()


# In[4]:


iris.hist()


# In[5]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris )
plt.show()


# In[6]:


sns.boxplot(x="Species", y="PetalWidthCm", data=iris )
plt.show()


# In[7]:


sns.boxplot(x="Species", y="SepalLengthCm", data=iris )
plt.show()


# In[8]:


sns.boxplot(x="Species", y="SepalWidthCm", data=iris )
plt.show()


# In[9]:



iris.plot(kind = 'box', subplots = True, layout= (2,2), sharex = False, sharey = False)
plt.show()


# In[10]:


from pandas.plotting import scatter_matrix
scatter_matrix(iris)
plt.show()


# In[ ]:




