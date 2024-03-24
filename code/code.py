import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
import numpy as np
import sklearn

#read data from file
missing_value = ['-9', 'NaN']
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\Data\Unprocessed_Data02.csv",skipinitialspace=True,na_values=missing_value)

#create a list to drop the redundant data
drop_columns = ['id','ccf', 'pncaden', 'htn', 'years', 'dm', 'exerwm', 'thalsev','thalpul', 'earlobe','diag','ramus',
               'om2','cathef','junk','restckm','exeref','exerckm','restwm','restef','famhist','name','lvf','lvx1','lvx2','lvx3','lvx4'
               ,'rldv5','ekgmo','ekgyr','ekgday','dig','prop','nitr','pro','diuretic','proto','met','thalrest','tpeakbps'
               ,'dummy','trestbpd','xhypo','rldv5e','cmo','cday','c','lmt','ladprox','laddist','cxmain','om1','rcaprox'
               ,'rcadist','tpeakbpd','painloc','painexer','relrest']

for col in drop_columns:
    if col in data:
        data.drop(col, axis=1, inplace=True)

#drop duplicate data
data.drop_duplicates(subset=None, keep='first',inplace=True)

#remove outlier
num_columns = data.loc[:, ~data.columns.isin(['sex','cp','smoke','fbs','restecg','exang','ca','thal','num'])]

data_summary = data[num_columns.columns].describe()

quartile_1 = data_summary.loc['25%']
quartile_3 = data_summary.loc['75%']
IQR = quartile_3 - quartile_1
range = 1.5 *IQR

noisy_data =[]
for col in num_columns:
    lower_boundaries = quartile_1[col] - range[col]
    upper_boundaries = quartile_3[col] + range[col]
    noisy_data += data.index[(data[col] < lower_boundaries) | (data[col] > upper_boundaries)].tolist()

data = data.drop(index=noisy_data)

#check and drop invalid data
sex_query = ((data['sex'] >=3) | (data['sex'] <= -1))
cp_query = ((data['cp']>= 5) | (data['age'] <= 0))
smoke_query = ((data['smoke']>= 2) | (data['smoke'] <= -1))
fbs_query = ((data['fbs']>= 2) | (data['fbs'] <= -1))
restecg_query = ((data['restecg']>= 4) | (data['restecg'] <= -1))
exang_query = ((data['exang']>= 2) | (data['exang'] <= -1))
slope_query = ((data['slope']>= 4) | (data['slope'] <= -1))
ca_query = ((data['ca']>= 4) | (data['ca'] <= -1))
thal_query = ((data['thal'] >= 8) | (data['thal'] <= 2))
num_query = ((data['num'] >= 2) | (data['num'] <= -1))

data = data[~sex_query]
data = data[~cp_query]
data = data[~smoke_query]
data = data[~fbs_query]
data = data[~restecg_query]
data = data[~exang_query]
data = data[~slope_query]
data = data[~ca_query]
data = data[~thal_query]
data = data[~num_query]

#Fill in data with KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(data)
data = pd.DataFrame(imputed, columns= data.columns)

#Bining data(smoke)
label = ['0','1']

enco = preprocessing.LabelEncoder()
data['smoke'] = enco.fit_transform(data['smoke'].astype(str))
data['smoke'] = pd.qcut(data['smoke'], q=2, labels=label, duplicates='drop')

unormalized_data = data.copy()

#normalized all columns
from sklearn.preprocessing import MinMaxScaler

normalize_columns = ['sex', 'smoke','cigs','fbs','restecg','exang','oldpeak','thal','num']
scaler = MinMaxScaler()
data[normalize_columns] = scaler.fit_transform(data[normalize_columns])

#Visualize duplicated data with histogram
flg, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,14))
unormalized_data[unormalized_data.columns].hist(column=data.columns, bins=5, ax=ax[0])
ax[0].set_title('Duplicate Rows')
plt.show()

#Visualize duplicated data with histogram

flg, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,14))
data[data.columns].hist(column=data.columns, bins=5, ax=ax[0])
ax[0].set_title('Duplicate Rows')
plt.show()


# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from numpy import mean

#import data

df =pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\assignmentData.csv")

# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# values to evaluate
over_values = [0.9]
under_values = [0.5]
for o in over_values:
  for u in under_values:
    # define pipeline
    model = SVC()
    over = SMOTE(sampling_strategy=o)
    under = RandomUnderSampler(sampling_strategy=u)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=5, n_jobs=-1)
    score = mean(scores)
    print('SMOTE oversampling rate:%.1f, Random undersampling rate:%.1f , Mean ROC AUC: %.3f' % (o, u, score))



#PART 1
# import SMOTE oversampling and other necessary libraries 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import data
df = pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\assignmentData.csv")


# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

#PART 2
# import SVM libraries 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

model=SVC()
clf_SMOTE = model.fit(X_train_SMOTE, y_train_SMOTE)
pred_SMOTE = clf_SMOTE.predict(X_test)



oversampled_data = pd.concat([X_train_SMOTE, y_train_SMOTE], axis=1)
oversampled_data.to_csv("oversampled_data.csv", index=False)
   


print("ROC AUC score for oversampled SMOTE data: ", roc_auc_score(y_test, pred_SMOTE))

#PART 1
# import random undersampling and other necessary libraries 
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import data
df =pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\assignmentData.csv")


# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

# summarize class distribution
print("Before undersampling: ", Counter(y_train))

# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

# summarize class distribution
print("After undersampling: ", Counter(y_train_under))

#PART 2
# import SVM libraries 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

model=SVC()
clf_under = model.fit(X_train_under, y_train_under)
pred_under = clf_under.predict(X_test)

print("ROC AUC score for undersampled data: ", roc_auc_score(y_test, pred_under))
undersampled_data = pd.concat([X_train_under, y_train_under], axis=1)
undersampled_data.to_csv("undersampled_data.csv", index=False)


#PART 1
# import sampling and other necessary libraries 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#import data
df =pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\Name1.csv")

# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# define pipeline
model = SVC()
over = SMOTE(sampling_strategy=0.7)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under), ('model', model)]
pipeline = Pipeline(steps=steps)

#PART 2
# import libraries for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from numpy import mean

# evaluate pipeline
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=5, n_jobs=-1)
score = mean(scores)
print('ROC AUC score for the combined sampling method: %.3f' % score)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

resampled_data = pipeline.named_steps['o'].fit_resample(X, y)
resampled_df = pd.DataFrame(data=resampled_data[0], columns=X.columns)
resampled_df['Class'] = resampled_data[1]
resampled_df.to_csv('resampled_datacombinationOver0.7Under1.csv', index=False)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="white",color_codes=True)

heart_disease=pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\resampled_datacombinationOver0.7Under1.csv")

print(heart_disease.head())

"""
print(heart_disease.isnull().sum())
#print out counts of each class value (0 , 1)
print(heart_disease["num"].value_counts())


#Check heart disease patient between Male and Female
sns.countplot(x='num',data=heart_disease,hue='sex')
# set plot title and axis labels
plt.title('Countplot of Heart Disease Patients by Sex')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')

# add a legend to the plot
plt.legend(title='Sex', loc='upper right', labels=['Female', 'Male'])
plt.show()




#plot histrogram for age
heart_disease['age'].plot.hist(bins=10)
plt.title('Age distribution for participant')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()



# filter the data for class attribute 1
class_1_data = heart_disease[heart_disease['num'] == 1]

# plot histogram of age for class attribute 1 patients
class_1_data['age'].plot.hist(bins=10)

# set plot title and axis labels
plt.title('Age distribution for participant who having heart disease')
plt.xlabel('Age')
plt.ylabel('Frequency')

# show the plot
plt.show()

# filter the data for class attribute 0
class_1_data = heart_disease[heart_disease['num'] == 0]

# plot histogram of age for class attribute 0 patients
class_1_data['age'].plot.hist(bins=10)

# set plot title and axis labels
plt.title('Age distribution for participant who not having heart disease')
plt.xlabel('Age')
plt.ylabel('Frequency')

# show the plot
plt.show()



sns.countplot(x='num',data=heart_disease,hue='cp')
plt.title('Countplot of Heart Disease Patients by Chest Pain Type')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')
# add a legend to the plot
#plt.legend(title='Chest Pain Type', loc='upper left', labels=['Typical angina', 'Atypical angina','Non-anginal pain','Asymptomatic'])
plt.show()




#plot histrogram for trestbps
heart_disease['trestbps'].plot.hist(bins=10)
plt.title('Resting blood pressure (mm Hg) distribution for participant')
plt.xlabel('Resting blood pressure (mm Hg)')
plt.ylabel('Frequency')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create separate dataframes for each group
hd = heart_disease[heart_disease['num'] > 0]
no_hd = heart_disease[heart_disease['num'] == 0]

# Plot histogram for each group
sns.histplot(hd['trestbps'], kde=True, label='Heart disease', color='red')
sns.histplot(no_hd['trestbps'], kde=True, label='No heart disease', color='blue')

# Add labels and legend
plt.xlabel('Resting blood pressure (mm Hg)')
plt.ylabel('Count')
plt.title('Resting blood pressure distribution for heart disease vs. no heart disease')
plt.legend()

# Show plot
plt.show()



# filter the data for class attribute 1
class_1_data = heart_disease[heart_disease['num'] == 1]

# plot histogram of age for class attribute 1 patients
class_1_data['trestbps'].plot.hist(bins=15)

# set plot title and axis labels
plt.title('Resting blood pressure (mm Hg) distribution for participant who having heart disease')
plt.xlabel('Resting blood pressure (mm Hg)')
plt.ylabel('Frequency')

# show the plot
plt.show()

# filter the data for class attribute 1
class_1_data = heart_disease[heart_disease['num'] == 1]

# plot histogram of age for class attribute 1 patients
class_1_data['chol'].plot.hist(bins=15)

# set plot title and axis labels
plt.title('Serum cholestoral in mg/dl distribution for participant who having heart disease')
plt.xlabel('Serum cholestoral (mg/dl)')
plt.ylabel('Frequency')

# show the plot
plt.show()


#plot histrogram for Serum cholestoral
heart_disease['chol'].plot.hist(bins=10)
plt.title('Serum cholestoral in mg/dl  distribution for participant')
plt.xlabel('Serum cholestoral in mg/dl ')
plt.ylabel('Frequency')
plt.show()



#Check heart disease patient between non smoker and smoker
sns.countplot(x='num',data=heart_disease,hue='smoke')
# set plot title and axis labels
plt.title('Countplot of Heart Disease Patients by smoking habit')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')

# add a legend to the plot
plt.legend(title='Smoker?', loc='upper right', labels=['Non smoker', 'Smoker'])
plt.show()





# create a cross-tabulation table
fbs_hd = pd.crosstab(heart_disease['smoke'], heart_disease['num'])

# rename the columns
fbs_hd.columns = ['Non smoker', 'Smoker']

# display the cross-tabulation table
print(fbs_hd)


import matplotlib.ticker as ticker
# normalize fbs values to 0 and 1
heart_disease['fbs'] = heart_disease['fbs'].apply(lambda x: 1 if x > 0.5 else 0)

# plot the countplot
sns.countplot(x='num', data=heart_disease, hue='fbs')

# set plot title and axis labels
plt.title('Countplot of Heart Disease Patients by fasting blood sugar')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')


# add a legend to the plot
plt.legend(title='Fasting Blood Sugar', loc='upper right', labels=['<120 mg/dl', '>120 mg/dl'])
plt.show()

# create a cross-tabulation table
fbs_hd = pd.crosstab(heart_disease['fbs'], heart_disease['num'])

# rename the columns
fbs_hd.columns = ['No Heart Disease', 'Heart Disease']

# display the cross-tabulation table
print(fbs_hd)




import seaborn as sns

sns.boxplot(x='num', y='chol', data=heart_disease)
plt.title('Serum Cholesterol and Heart Disease')
plt.xlabel('Heart Disease Diagnosis')
plt.ylabel('Serum Cholesterol (mg/dl)')
plt.show()


import matplotlib.pyplot as plt

plt.hist(heart_disease[heart_disease['num']==1]['chol'], alpha=0.5, label='Heart Disease')
plt.hist(heart_disease[heart_disease['num']==0]['chol'], alpha=0.5, label='No Heart Disease')
plt.title('Serum Cholesterol and Heart Disease')
plt.xlabel('Serum Cholesterol (mg/dl)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# plot the countplot
sns.countplot(x='num', data=heart_disease, hue='restecg')
# set plot title and axis labels
plt.title('Countplot of Heart Disease Patients by resting electrocardiographic results')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')


# add a legend to the plot
plt.legend(title='Resting electrocardiographic results', loc='upper right', labels=['0:normal', '1:having ST-T wave abnormality','2:showing probable'])
plt.show()


#plot histrogram for Maximum heart rate
heart_disease['thalach'].plot.hist(bins=10)
plt.title('Maximum heart rate distribution for participants')
plt.xlabel('Maximum heart rate')
plt.ylabel('Frequency')
plt.show()
# filter the data for class attribute 1
class_1_data = heart_disease[heart_disease['num'] == 1]

# plot histogram of age for class attribute 1 patients
class_1_data['thalach'].plot.hist(bins=15)

# set plot title and axis labels
plt.title('Maximum heart rate distribution for participant who having heart disease')

plt.xlabel('Maximum heart rate')
plt.ylabel('Frequency')


# show the plot
plt.show()



plt.hist(heart_disease[heart_disease['num']==1]['thalach'], alpha=0.5, label='Heart Disease')
plt.hist(heart_disease[heart_disease['num']==0]['thalach'], alpha=0.5, label='No Heart Disease')
plt.title('Maximum heart rate achieved and Heart Disease')
plt.xlabel('Maximum heart rate')
plt.ylabel('Frequency')
plt.legend()
plt.show()


sns.boxplot(x='num', y='thalach', data=heart_disease)
plt.title('Maximum heart rate and Heart Disease')
plt.xlabel('Heart Disease Diagnosis')
plt.ylabel('Maximum heart rate')
plt.show()


# normalize fbs values to 0 and 1
heart_disease['exang'] = heart_disease['exang'].apply(lambda x: 1 if x > 0.5 else 0)


#Check heart disease patient between exang
sns.countplot(x='num',data=heart_disease,hue='exang')
# set plot title and axis labels
plt.title('Countplot of Participants by exercise induced angina ')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')

# add a legend to the plot
plt.legend(title='exercise induced angina ', loc='upper right', labels=['No(0)', 'Yes(1)'])
plt.show()


# round up slope values

heart_disease['slope'] = heart_disease['slope'].apply(lambda x: round(x))

# remove 0 category from slope
heart_disease = heart_disease[heart_disease['slope'] != 0]


# create countplot
sns.countplot(x='num', data=heart_disease, hue='slope')

# set plot title and axis labels
plt.title('Countplot of Participants by slope of the peak exercise ST segment')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Count')

# add a legend to the plot
plt.legend(title='Slope of the peak exercise ST segment', loc='upper right', labels=['upsloping', 'flat', 'downsloping'])

# show the plot
plt.show()



# create boxplot
sns.boxplot(x='num', y='oldpeak', data=heart_disease)

# set plot title and axis labels
plt.title('Boxplot of ST depression by diagnosis of heart disease')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('ST depression induced by exercise relative to rest')

# show the plot
plt.show()

"""

#plot histrogram for ca
heart_disease['ca'].plot.hist(bins=10)
plt.title('Histrogram of number of major vessels colored by flourosopy for participant')
plt.xlabel('Number of major vessels colored by flourosopy')
plt.ylabel('Frequency')
plt.show()

# create boxplot
sns.boxplot(x='num', y='ca', data=heart_disease)

# set plot title and axis labels
plt.title('Boxplot of number of major vessels (0-3) colored by flourosopy by diagnosis of heart disease')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Number of major vessels colored by flourosopy')

# show the plot
plt.show()

plt.hist(heart_disease[heart_disease['num']==1]['ca'], alpha=0.5, label='Heart Disease')
plt.hist(heart_disease[heart_disease['num']==0]['ca'], alpha=0.5, label='No Heart Disease')
plt.title('Number of major vessels (0-3) colored by flourosopy and Heart Disease')
plt.xlabel('Number of major vessels (0-3) colored by flourosopy')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# create boxplot
sns.boxplot(x='num', y='ca', data=heart_disease)

# set plot title and axis labels
plt.title('Boxplot of number of major vessels (0-3) colored by flourosopy by diagnosis of heart disease')
plt.xlabel('Diagnosis of heart disease')
plt.ylabel('Number of major vessels colored by flourosopy')

# show the plot
plt.show()


