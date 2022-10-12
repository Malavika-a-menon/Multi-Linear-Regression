#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv(r'C:/Users/Admin/Downloads/archive (1) (1)/Fish.csv')
data.head()


# In[7]:


x = data[['Width','Weight']]
y = data['Height']


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression
#creating an object of linearregression class
LR= LinearRegression()
#fitting the training data
LR.fit(x_train,y_train)


# In[13]:


y_pred=LR.predict(x_test)
print(y_pred)


# In[14]:


LR.predict([(14.96,6.7)])


# In[15]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[16]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('actual VS Predictd')


# In[17]:


pred_y_df = pd.DataFrame({'Actual Value':y_test,'Predicted value':y_pred,'Difference':y_test-y_pred})
pred_y_df[0:10]


# In[ ]:




