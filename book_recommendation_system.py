# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:31:39 2022

@author: Mohd Ariz Khan
"""

# Import thr data
import pandas as pd
df = pd.read_csv("book.csv", encoding = 'latin1')
df

# Get information of the datasets
df.info()
print('The shape of our data is:', df.shape)
df.isnull().any()

df.sort_values('User_ID')

# Number of unique users in the dataset
len(df)
len(df.User_ID.unique())


#=============================================================================
#                      EDA (Exploratory Data Analysis)
#=============================================================================
df['Book_Rating'].value_counts()

# Histogram
df['Book_Rating'].hist()

len(df.Book_Title.unique())
df.Book_Title.value_counts()
t1 = df.Book_Title.value_counts()

# Bar graph
t1.plot(kind='bar')

# Use Pivot Table to reshape the data
user_df = df.pivot_table(index ='User_ID', columns='Book_Title', values='Book_Rating')

print(user_df)

# Impute those NaNs with 0 values 
user_df.fillna(0, inplace=True)
user_df

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric = 'cosine')
user_sim

# Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

# Set the index and column names to user ids 
user_sim_df.index   = df.User_ID.unique()
user_sim_df.columns = df.User_ID.unique()

user_sim_df

# select first five users
user_sim_df.iloc[:5,:5]

import numpy as np
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:7, 0:7]

# To save your cosin calcutaion file
user_sim_df.to_csv("cosin_calc.csv")

# Most Similar Users
user_sim_df.max()
user_sim_df.idxmax(axis=1)[0:10]


df[(df['User_ID'] == 276729) | (df['User_ID'] == 276726)]

# System will recommend 'Classical Mythology' to user_276729 and 'Decision in Normandy' & 'Clara Callan' to user_276726
user_276729 = df[df['User_ID'] == 276729]
user_276726 = df[df['User_ID']  == 276726]

pd.merge(user_276729,user_276726,on='Book_Title', how='inner')
pd.merge(user_276729,user_276726,on='Book_Title',how='outer')


df[(df['User_ID'] == 276737) | (df['User_ID'] == 276726)]

# System will recommend 'Classical Mythology' to user_276737 and 'The Mummies of Urumchi' to user_276726
user_276737 = df[df['User_ID'] == 276737]
user_276726 = df[df['User_ID']  == 276726]

pd.merge(user_276737,user_276726,on='Book_Title', how='inner')
pd.merge(user_276737,user_276726,on='Book_Title',how='outer')


df[(df['User_ID'] == 276744) | (df['User_ID'] == 276726)]

# System will recommend 'Classical Mythology' to user_276744 and 'The Kitchen God's Wife' & to user_276726
user_276744 = df[df['User_ID'] == 276744]
user_276726 = df[df['User_ID']  == 276726]

pd.merge(user_276744,user_276726,on='Book_Title', how='inner')
pd.merge(user_276744,user_276726,on='Book_Title',how='outer')

#==========================================================================================================

