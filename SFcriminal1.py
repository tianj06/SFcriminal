# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

train_data = pd.read_csv('C:/Users/jutian/Documents/GitHub/SFcriminal data/train.csv')

train_data.head()
pd.unique(train_data['Category'])
crime_counts = pd.value_counts(train_data['Category'])

# plot top 20 most frequent crimes
ypos = np.arange(20)
plt.figure()
plt.barh(ypos,crime_counts[:20][::-1])
plt.yticks(ypos, crime_counts.index[:20][::-1])
plt.xlabel('crime counts')
plt.title('top 20 crimes')

# plot cumulative probably of all types of crimes
cumsum_crime = np.cumsum(crime_counts)/crime_counts.sum()
plt.figure()
plt.plot(cumsum_crime)
plt.vlines([5,10,15],0,1, colors='r', linestyles='dashed')
for i in [5,10,15]:
    plt.text(i,cumsum_crime[i],str(cumsum_crime[i]))
    
# plot distribution of crime for each district
    
def dist_by_group(grp, predictor):
    df = pd.concat([grp, predictor], axis=1) 
    colnames = df.columns.values
    grouped = df.groupby(colnames[0])
    agg_df = grouped.apply(lambda x: pd.value_counts(x.iloc[:,1])/sum(pd.value_counts(x.iloc[:,1])))
    
pd.unique(train_data['PdDistrict'])


train_data.describe()

# convert criminal into indicators 
# determine all of the crime types:
crimes = set()
for m in train_data.Category:
    crimes.update re.split('/',m)
criminalType = train_data['Category'].apply(re.split('/',)) 