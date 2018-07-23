## Challenge 1
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats

# Load data
data = pd.read_excel('challenge/data/ccrb_datatransparencyinitiative_20170207.xlsx','Complaints_Allegations')
data_removena = data.dropna()
data_unique = data_removena.groupby('UniqueComplaintId').first()
data_withna_unique = data.groupby('UniqueComplaintId').first()

""" 1. Count the unique complaints with complete information """
n_unique_complete = len(data_unique)
print("N of unique complaints with complete information: {}".format(n_unique_complete))

""" 2. Proportion of complaints in the borough with largest N of complaints """
count_for_each_borough = data_unique.groupby('Borough of Occurrence')['Borough of Occurrence'].count()
borough_max_count = count_for_each_borough.argmax()
count_in_this_borough = np.sum(data_unique['Borough of Occurrence']==borough_max_count)
proportion_in_this_borough = count_in_this_borough/n_unique_complete
print("Proportion of complaints in the borough with largest N of complaints: {}".format(proportion_in_this_borough))

""" 3. Complaints per 100k residents in borough with largest N of complaints per capita """
Population_2016 = {'Manhattan':1643734,'Brooklyn':2629150,'Queens':2333054,'Bronx':1455720,'Staten Island':476015}
count_per_capita = {}
for key in Population_2016.keys():
    count_per_capita[key] = count_for_each_borough[key]/Population_2016[key]
max_count_per_100k = max(count_per_capita.values())*100000
print("Complaints per 100k residents in borough with largest N of complaints per capita: {}".format(max_count_per_100k))

""" 4. Averaged number of years taken for closing a complaint """
years_taken = data_unique['Close Year'] - data_unique['Received Year']
mean_years_taken = np.mean(years_taken)
print("Averaged number of years taken for closing a complaint: {}".format(mean_years_taken))

""" 5. Predict number of complaints about stop and frisk in 2018 using linear regression """
sf_by_year = data_unique.groupby('Incident Year')['Complaint Contains Stop & Frisk Allegations'].sum()
x = sf_by_year.index.values
y = np.array(sf_by_year)
x, y = (x[np.logical_and(2007<=x, x<=2016)], y[np.logical_and(2007<=x, x<=2016)])
model = linear_model.LinearRegression()
model.fit(x[:, None], y)
pred_2018 = model.predict(2018)
print("Predicted number of complaints about stop and frisk in 2018: {}".format(pred_2018))

""" 6. Chi-square test for whether having video increases the chance of a complaint getting full investigation """
grouped_video_full = np.array(data_unique.groupby(['Complaint Has Video Evidence','Is Full Investigation'])['DateStamp'].count())
RC_table = grouped_video_full.reshape([2,2])
stat, pvalue, dof, expected = stats.chi2_contingency(RC_table)
print("Chi-square test statistic for whether having video increases the chance of a complaint getting full investigation: {}".format(stat))

""" 7. Correlation between number of alligations per complaint and the alligation type """
type_list = np.unique(data_removena['Allegation FADO Type'])
type_indicator = pd.get_dummies(data_removena['Allegation FADO Type'])
data_with_indicator = pd.concat([data_removena, type_indicator], axis=1)
X = data_with_indicator.groupby('UniqueComplaintId')[type_list].max()
Y = np.sum([X[type_i] for type_i in type_list], axis=0)
for i in range(type_indicator.shape[1]):
    model = linear_model.LinearRegression()
    model.fit(X[type_list[i]][:,None], Y)
    print('Coefficient for {} is {}'.format(type_list[i],model.coef_))

""" 8. Number of officers per precinct """
N_officer_total = 36000
precinct_number = {'Manhattan':22, 'Bronx':12, 'Brooklyn':23, 'Queens':16, 'Staten Island':4}
count_complaint_each_borough = data_withna_unique.groupby('Borough of Occurrence')['Borough of Occurrence'].count()
total_complaint_count = sum(count_for_each_borough)-count_for_each_borough['Outside NYC']
N_officer_per_precinct = []
for key in precinct_number.keys():
    N_officer = N_officer_total*count_complaint_each_borough[key]/total_complaint_count
    N_officer_per_precinct.append(N_officer/precinct_number[key])
ratio = max(N_officer_per_precinct)/min(N_officer_per_precinct)
print('Ratio of max N officer per precinct to min N officer per precinct is {}'.format(ratio))

## Challenge 2
import numpy as np

T = 5
N0 = 64
S = []
for i in range(1000000):
    N = N0
    for t in range(T):
        if N>2:
            cut_points = np.sort(np.random.choice(N-1,2,replace=False)+1)
            N = np.max([cut_points[0],cut_points[1]-cut_points[0],N-cut_points[1]])
    S.append(N)
print('Mean of S when N=64 and T=5: {}'.format(np.mean(S)))
print('Standart deviation of S when N=64 and T=5: {}'.format(np.std(S)))
print('Conditional probability of S>8 given S>4: {}'.format(np.sum(np.array(S)>8)/np.sum(np.array(S)>4)))

T = 10
N0 = 1024
S = []
for i in range(1000000):
    N = N0
    for t in range(T):
        if N>2:
            cut_points = np.sort(np.random.choice(N-1,2,replace=False)+1)
            N = np.max([cut_points[0],cut_points[1]-cut_points[0],N-cut_points[1]])
    S.append(N)
print('Mean of S when N=64 and T=5: {}'.format(np.mean(S)))
print('Standart deviation of S when N=64 and T=5: {}'.format(np.std(S)))
print('Conditional probability of S>8 given S>4: {}'.format(np.sum(np.array(S)>8)/np.sum(np.array(S)>4)))