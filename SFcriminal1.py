# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA 
#import sklearn.metrics.classification_report as class_repo
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
#from pandas.tools.plotting import parallel_coordinates


data = pd.read_csv('C:/Users/jutian/Documents/GitHub/SFcriminal data/train.csv')
test = pd.read_csv('C:/Users/jutian/Documents/GitHub/SFcriminal data/test.csv')

data.head()
pd.unique(data['Category'])
crime_counts = pd.value_counts(data['Category'])

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
    agg_df = agg_df.unstack()
    return agg_df

crime_ratio_District = dist_by_group(data.PdDistrict, data.Category)
plt.figure()
topCrimeByDistrict = crime_ratio_District[crime_counts.index[:5]]
topCrimeByDistrict.plot()
# clearly district has quite a lot of information about different crimes
plt.figure()
crime_ratio_Day =  dist_by_group(data.DayOfWeek, data.Category)
topCrimeByDay = crime_ratio_Day[crime_counts.index[:5]]
topCrimeByDay = topCrimeByDay.reindex(index=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
topCrimeByDay.plot()

# just build a crapy model first using DayofWeek, PdDistrict as features

X = pd.get_dummies(data[['PdDistrict','DayOfWeek']])
X.drop(['DayOfWeek_Monday','PdDistrict_BAYVIEW'], axis=1, inplace=True)
VarNames = list(X.columns.values)
X = X.values
y, labels = pd.factorize(data['Category'])

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf1 = LDA().fit(X_train, y_train)

pred = clf1.predict(X_test)

accuracy_score(y_test, pred, normalize=True)
#class_repo(testy, pred, target_names = VarNames)



#%% last bit
testX = pd.get_dummies(test[['PdDistrict','DayOfWeek']])
testX = testX[VarNames]
testX = testX.values
testy, testlabels = pd.factorize(test['Category'])

# just make sure that test data and train data are factorized in the same way
np.array_equal(labels,testlabels)