#!/usr/bin/env python
# coding: utf-8

# In[77]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from subprocess import check_output


# In[78]:


data=pd.read_csv('C:/Users/NK(The Bad Boy)/Downloads/Compressed/archive/data.csv')


# In[79]:


data.info()


# In[80]:


data.head()


# In[81]:


data = data.drop('id',axis=1)
data = data.drop('Unnamed: 32',axis=1)


# In[82]:


data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})


# In[83]:


datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))
datas.columns = list(data.iloc[:,1:32].columns)
datas['diagnosis'] = data['diagnosis']


# In[84]:


data_drop = datas.drop('diagnosis',axis=1)
X = data_drop.values


# In[85]:


from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
Y = tsne.fit_transform(X)


# In[86]:


from sklearn.cluster import KMeans


# In[87]:


kmns = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kY = kmns.fit_predict(X)
kY


# In[88]:


plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.scatter(X[kY == 0,22], X[kY == 0,27], s = 50, c = 'green' , marker = '', edgecolor = 'black', label = 'cluster 0')
plt.scatter(X[kY == 1,22], X[kY == 1,27], s = 50, c = 'orange' , marker = '.', edgecolor = 'black', label = 'cluster 1')

plt.legend()
plt.title('K Means')
plt.grid()
plt.subplot(1,2,2)
plt.scatter(X[:,22],X[:,27], cmap = 'rainbow')
plt.title('original')
plt.show()


# In[101]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "rainbow", marker = 'o', edgecolor = "black", alpha=0.35)
ax1.set_title('k-means clustering plot')
ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "rainbow", edgecolor = "black", alpha=0.35)
ax2.set_title('Actual clusters')


# In[ ]:




