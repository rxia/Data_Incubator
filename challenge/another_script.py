## Challenge 1
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
import sklearn
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel('challenge/data/ccrb_datatransparencyinitiative_20170207.xlsx','Complaints_Allegations')

## Q1
data.head(5)

# print unique values of every col
for col in data:
    print(col)
    print(data[col].unique())

# the missing values simiply contains NaN

print(len(data))
data_complete = data.dropna(axis=0)
print(len(data_complete))

data_complete_unique = data_complete.groupby('UniqueComplaintId').first()

num_unique_id = len(data_complete['UniqueComplaintId'].unique())

print(num_unique_id)



## Q2
data_complete['count'] = 1

complain_by_borough = data_complete[['UniqueComplaintId', 'Borough of Occurrence', 'count']]\
    .groupby('UniqueComplaintId').first().groupby('Borough of Occurrence').count()

print(count_by_borough)

print('{:0.10f}'.format((complain_by_borough['count'].max()/complain_by_borough['count'].sum())))


## Q3


complain_by_borough['population'] = 0
complain_by_borough['population']['Bronx'] = 1455720
complain_by_borough['population']['Brooklyn'] = 2629150
complain_by_borough['population']['Manhattan'] = 1643734
complain_by_borough['population']['Queens'] = 2333054
complain_by_borough['population']['Staten Island'] = 476015

complain_by_borough['complain_per_100k'] = complain_by_borough['count']* 10**5 / complain_by_borough['population']

complain_by_borough


## 04

dur_by_id = data_complete[['UniqueComplaintId', 'Close Year', 'Received Year']]\
    .groupby('UniqueComplaintId').first()

print(np.mean(dur_by_id['Close Year'] - dur_by_id['Received Year']))


## 05

sf_by_year = data_complete_unique.groupby('Incident Year')['Complaint Contains Stop & Frisk Allegations'].sum()

print(sf_by_year)

x = sf_by_year.index.values
y = np.array(sf_by_year)

x, y = (x[np.logical_and(2007<=x, x<=2016)], y[np.logical_and(2007<=x, x<=2016)])

model = sklearn.linear_model.LinearRegression()
model.fit(x[:, None], y)
y_2018 = model.predict([(2018, )])

print(y_2018)
# 232.60606061


plt.plot(x, y)
plt.plot(2018, y_2018, 'o')


## 06

keys = ['Is Full Investigation', 'Complaint Has Video Evidence']

temp = data_complete_unique.groupby(['Is Full Investigation', 'Complaint Has Video Evidence']).size()

print(temp)

contingency_table = [[44529, 584], [21889, 1465]]

stats.chi2_contingency(contingency_table, )


## 07

types_allegation = data_complete['Allegation FADO Type'].unique()
x = data_complete.groupby('UniqueComplaintId')['Allegation FADO Type']\
    .apply(lambda x: np.any(x[:, None]==types_allegation[None, :], axis=0))

x = np.array(x.tolist(), dtype='float')
y = np.sum(x, axis=1)

model = sklearn.linear_model.LinearRegression()
for i in range(x.shape[1]):
    model.fit(x[:, i:i+1], y)
    print(model.coef_)

"""
[ 0.87614115]
[ 0.59770152]
[ 0.66306306]
[ 1.01520846]
"""

## 08


df_complain = pd.DataFrame({'num_complain': data.groupby('UniqueComplaintId').first().groupby('Borough of Occurrence').size()})
df_complain['num_precinct'] = 0
df_complain['num_precinct']['Bronx'] = 12
df_complain['num_precinct']['Brooklyn'] = 23
df_complain['num_precinct']['Manhattan'] = 22
df_complain['num_precinct']['Queens'] = 16
df_complain['num_precinct']['Staten Island'] = 4

complain_per_percint = 1.0*df_complain['num_complain']/df_complain['num_precinct']
complain_per_percint['Bronx'] / complain_per_percint['Queens']



