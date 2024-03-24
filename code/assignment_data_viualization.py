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
