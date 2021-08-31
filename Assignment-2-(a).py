#!/usr/bin/env python
# coding: utf-8

# In[176]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[177]:



from sklearn.datasets import load_breast_cancer


# In[178]:


cancer=load_breast_cancer()


# In[202]:


cancer.keys()


# In[203]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head(5)


# In[204]:


from sklearn.preprocessing import MinMaxScaler


# In[205]:


from sklearn.preprocessing import StandardScaler


# In[206]:


scaler=StandardScaler()
scaler.fit(df)


# In[207]:


scaled_data=scaler.transform(df)


# In[208]:


scaled_data


# In[209]:


from sklearn.decomposition import PCA


# In[210]:


pca=PCA(n_components=3)


# In[211]:


pca.fit(scaled_data)


# In[212]:


x_pca=pca.transform(scaled_data)


# In[213]:


scaled_data.shape


# In[214]:



x_pca.shape


# In[215]:


scaled_data


# In[216]:



x_pca


# In[217]:



plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')

