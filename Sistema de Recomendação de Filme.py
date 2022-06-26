#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from fastai.vision.all import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
from PIL import Image
plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")


# In[2]:


movies_md = pd.read_csv("movies_metadata.csv")
movies_keywords = pd.read_csv("keywords.csv")
movies_credits = pd.read_csv("credits.csv")


# In[3]:


movies_md.head()


# In[4]:


movies_md.info()


# In[5]:


movies_keywords.head()


# In[6]:


movies_credits.head()


# In[7]:


movies_md = movies_md[movies_md['vote_count'] >= 55]


# In[8]:


movies_md.head()


# In[9]:


movies_md.columns


# In[10]:


movies_md = movies_md[['id', 'original_title', 'overview', 'genres']]


# In[11]:


movies_md.head()


# In[12]:


movies_md['title'] = movies_md['original_title']


# In[13]:


movies_md.head()


# In[14]:


movies_md.reset_index(inplace=True, drop=True)


# In[15]:


movies_md.head()


# In[16]:


movies_credits = movies_credits[['id', 'cast']]


# In[17]:


movies_credits.head()


# In[18]:


len(movies_md)


# In[19]:


movies_md = movies_md[movies_md['id'].str.isnumeric()]


# In[20]:


len(movies_md)


# In[21]:


movies_md['id'] = movies_md['id'].astype(int)


# In[22]:


movies_df = pd.merge(movies_md, movies_keywords, on='id', how='left')


# In[23]:


movies_df.head()


# In[24]:


movies_df.reset_index(inplace=True, drop=True)


# In[25]:


movies_df = pd.merge(movies_df, movies_credits, on='id', how='left')
movies_df.reset_index(inplace=True, drop=True)


# In[26]:


movies_df.head()


# In[27]:


movies_df['genre'] = movies_df['genres'].apply(lambda x: [i['name'] for i in eval(x)])


# In[28]:


movies_df.head()


# In[29]:


movies_df['genre'] = movies_df['genres'].apply(lambda x: [i['name'] for i in eval(x)])


# In[30]:


movies_df.head()


# In[31]:


movies_df['genre'] = movies_df['genre'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[32]:


movies_df.head()


# In[33]:


movies_df.isnull().sum()


# In[34]:


movies_df['keywords'].fillna('[]', inplace=True)


# In[35]:


movies_df['genre'] = movies_df['genre'].apply(lambda x: ' '.join(x))


# In[36]:


movies_df.head()


# In[37]:


movies_df.drop('genres', axis=1, inplace=True)


# In[38]:


movies_df.head()


# In[39]:


movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'] for i in eval(x)])
movies_df['cast'] = movies_df['cast'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))


# In[40]:


movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))


# In[41]:


movies_df.head()


# In[42]:


movies_df['tags'] = movies_df['overview']+' '+movies_df['keywords']+' '+movies_df['cast']+' '+movies_df['genre']+' '+movies_df['original_title']


# In[43]:


movies_df['tags']


# In[44]:


movies_df.drop(['genre', 'original_title', 'keywords', 'cast', 'overview'], axis=1, inplace=True)


# In[45]:


movies_df.head()


# In[46]:


movies_df.isnull().sum()


# In[47]:


movies_df.drop(movies_df[movies_df['tags'].isnull()].index, inplace=True)


# In[48]:


movies_df.shape


# In[49]:


movies_df.drop_duplicates(inplace=True)


# In[50]:


movies_df.shape


# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[52]:


tfidf = TfidfVectorizer(max_features=5000)


# In[53]:


vectorized_data = tfidf.fit_transform(movies_df['tags'].values)


# In[54]:


tfidf.get_feature_names_out()


# In[55]:


vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=movies_df['tags'].index.tolist())


# In[56]:


vectorized_dataframe.head()


# In[57]:


vectorized_dataframe.shape


# In[58]:


from sklearn.decomposition import TruncatedSVD


# In[59]:


svd = TruncatedSVD(n_components=3000)

reduced_data = svd.fit_transform(vectorized_dataframe)


# In[60]:


reduced_data.shape


# In[61]:


reduced_data


# In[62]:


svd.explained_variance_ratio_.cumsum()


# In[63]:


from sklearn.metrics.pairwise import cosine_similarity


# In[64]:


similarity = cosine_similarity(reduced_data)


# In[65]:


similarity


# In[66]:


def recomendation_system(movie):
    id_of_movie = movies_df[movies_df['title']==movie].index[0]
    distances = similarity[id_of_movie]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:15]
    for movie_id in movie_list:
        print(movies_df.iloc[movie_id[0]].title)


# In[67]:


recomendation_system('Rango')


# In[68]:


recomendation_system('Charlie and the Chocolate Factory')


# In[ ]:




