
# coding: utf-8

# # Bigmart Sales Analysis: For data comprising of transaction records of a sales store. 
# The data has 8523 rows of 12 variables. Predict the sales of a store. Sample Test data set available
# here https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/
# 
# Variable - Description
# 
# Item_Identifier - Unique product ID
# 
# Item_Weight - Weight of product
# 
# Item_Fat_Content - Whether the product is low fat or not
# 
# Item_Visibility - The % of total display area of all products in a store allocated to the particular product
# 
# Item_Type - The category to which the product belongs
# 
# Item_MRP - Maximum Retail Price (list price) of the product
# 
# Outlet_Identifier - Unique store ID
# 
# Outlet_Establishment_Year - The year in which store was established
# 
# Outlet_Size - The size of the store in terms of ground area covered
# 
# Outlet_Location_Type - The type of city in which the store is located
# 
# Outlet_Type - Whether the outlet is just a grocery store or some sort of supermarket
# 
# Item_Outlet_Sales - Sales of the product in the particulat store. This is the outcome variable to be predicted.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Train = pd.read_csv("/home/administrator/Downloads/Train.csv",header=None)
Test = pd.read_csv("/home/administrator/Downloads/Test.csv",header=None)


# In[3]:


headers = ['Item_Identifier','Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']


# In[4]:


Train.columns = headers
Test.columns = headers[:11]


# In[5]:


Train['Source'] = 'Train'
Test['Source'] = 'Test'


# In[6]:


Data = pd.concat([Train,Test],ignore_index=True,sort=True)


# normalise Item_Fat_Content

# In[7]:


Data['Item_Fat_Content'].replace('LF','Low',inplace = True)


# In[8]:


Data['Item_Fat_Content'].replace('low fat','Low',inplace = True)


# In[9]:


Data['Item_Fat_Content'].replace('reg','Regular',inplace = True)


# replace missing values with mean for Item_Weight

# In[10]:


Item_Weight_Mean = Data['Item_Weight'].mean(axis=0)


# In[11]:


Data['Item_Weight'].replace(np.NaN,Item_Weight_Mean, inplace = True)


# replace missing values with mean for Item_Visibility

# In[12]:


Data['Item_Visibility'].replace(0,np.NaN,inplace = True)


# In[13]:


Item_Visibility_Mean = Data['Item_Visibility'].mean(axis = 0)


# In[14]:


Data['Item_Visibility'].replace(np.NaN,Item_Visibility_Mean,inplace = True)


# replace item_type by itemID initials (to reduce total number of types from 16 to 3)

# In[15]:


Data['Item_Type'] = Data['Item_Identifier'].apply(lambda x : x[0:2])


# replace missing values for Outlet_Size

# In[16]:


from scipy.stats import mode


# In[17]:


Outlet_Size_mode = Data.pivot_table(values = 'Outlet_Size',columns = 'Outlet_Type', aggfunc = (lambda x:x.mode().iat[0]))


# In[18]:


miss_bool = Data['Outlet_Size'].isnull()


# In[19]:


Data.loc[miss_bool,'Outlet_Size'] = Data.loc[miss_bool,'Outlet_Type'].apply(lambda x: Outlet_Size_mode[x])


# Convert categorical to numerical using dummy columns

# In[20]:


dummies = ['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Size','Outlet_Type']


# In[21]:


Data = pd.get_dummies(Data, columns = dummies)


# Drop useless columns

# In[22]:


Data.drop(['Outlet_Identifier','Item_Identifier'],axis=1, inplace=True)


# split df into train and test

# In[25]:


Train  = Data.loc[Data['Source']=='Train']
Test = Data.loc[Data['Source']=='Test']


# In[26]:


Train.drop('Source', axis = 1, inplace = True)
Test.drop('Source', axis = 1, inplace = True)


# In[28]:


x_train = np.array(Train.drop(['Item_Outlet_Sales'],axis=1))
y_train = np.array(Train['Item_Outlet_Sales'])


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[30]:


lr = LinearRegression(normalize = True)


# In[31]:


lr.fit(x_train,y_train)


# In[32]:


lr.intercept_


# In[33]:


lr.coef_


# In[41]:


y_train_pred = lr.predict(x_train)


# In[42]:


rmse = metrics.mean_squared_error(y_train,y_train_pred)


# In[43]:


rmse


# In[44]:


y_train_pred

