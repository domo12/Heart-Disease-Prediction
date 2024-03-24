import pandas as pd
import numpy as np
import statsmodels.api as sm

#explore the relationship between age and the incidence of heart disease and
#the difference in incidence between males and female using logistic regression

# Load the heart disease dataset
df = pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\resampled_datacombinationOver0.7Under1.csv")

#create binary variables for sex and class(heart_disease) columns
df['Class'] = np.where(df['Class']>0, 1, 0)
df['sex'] = np.where(df['sex']==1, 1, 0)

X = df[['age', 'sex','cp','trestbps','chol','smoke','cigs','fbs','restecg','thaldur','thaltime','thalach','exang','oldpeak','slope','ca','thal']]
y = df['Class']
#show regression coefficients and odds ratios for the 'age' and 'sex' 
X = sm.add_constant(X)


logit_model = sm.Logit(y, X)
results = logit_model.fit()
print(results.summary())

df['predicted_prob'] = results.predict(X)

print(df['predicted_prob'])



#decision tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns





#Preparing data for training and test
#Define feature vector and target variable (class)
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'smoke', 'cigs', 'fbs', 'restecg', 'thaldur', 'thaltime', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df ['Class']
print (X.head ())


#Split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split (X, y, \
                            test_size = 0.33, random_state = 42)

#Check the shape of X_train and X_test
print ("\n")
print("train and test sample size", X_train. shape, X_test.shape)


#Import and instantiate the DecisionTreeClassifier model with entropy
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier (criterion='entropy', max_depth=4)

#Fit the decision tree model
dt = dt.fit (X_train, y_train)



#Let's visualize the decision tree
from sklearn import tree
features= ['age', 'sex','cp','trestbps','chol','smoke','cigs','fbs','restecg','thaldur','thaltime','thalach','exang','oldpeak','slope','ca','thal']
class_names = ['0', '1']
#tree.plot_tree (dt, feature_names=features, class_names=class_names)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(12,8), dpi=100)
tree.plot_tree(dt, feature_names=features, class_names=class_names, filled=True, ax=axes);
#Predict the Test set results
y_pred = dt.predict (X_test)


# Creates a confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix (y_test, y_pred)
print ("---------------\n")
print(cm)
       
#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')




plt.show()


#SVM and RBF

from sklearn.svm import SVC
from sklearn.metrics import classification_report
svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("\n SVM Evaluation :\n")
print(classification_report(y_test,y_pred, zero_division=0))


print("\n SVM Confusion matrix :\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('Linear SVM Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
svm=SVC(kernel='rbf')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("\nSVM RBF Evaluation:\n")
print(classification_report(y_test,y_pred, zero_division=0))

print("\n SVM RBF Confusion matrix :\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)

    
#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('RBF SVM Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')




plt.show()

#k nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

k_values=[i for i in range(10,31)]
scores=[]
error_rate=[]

for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(knn.score(X_test,y_test))
    error_rate.append(np.mean(y_pred!=y_test))


sns.lineplot(x=k_values, y=scores,marker='o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()


sns.lineplot(x=k_values, y=error_rate,marker='o')
plt.xlabel("K Values")
plt.ylabel("Error Rate")
plt.show()

#Get best k and train kNN

best_index=np.argmax(scores)
best_k=k_values[best_index]

knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("\nk = ",best_k)
print("\nkNN Evaluation:\n")
print(classification_report(y_test,y_pred))

print("\nkNN confusion matrix:\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)
    
#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('kNN Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
#explore the relationship between age and the incidence of heart disease and
#the difference in incidence between males and female using logistic regression

# Load the heart disease dataset
df = pd.read_csv(r"C:\Users\wong\OneDrive\Documents\Degree\Y2S3\UECS3213  UECS3453  UECS3483 DATA MINING\resampled_datacombinationOver0.7Under1.csv")






#Preparing data for training and test
#Define feature vector and target variable (class)
X = df[['age', 'cp','thaltime','thal','oldpeak','slope','thaldur','cigs']]

y = df ['Class']
print (X.head ())


#Split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split (X, y, \
                            test_size = 0.33, random_state = 42)





#SVM and RBF

from sklearn.svm import SVC
from sklearn.metrics import classification_report
svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("\n SVM Evaluation :\n")
print(classification_report(y_test,y_pred, zero_division=0))


print("\n SVM Confusion matrix :\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('Linear SVM Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
svm=SVC(kernel='rbf')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("\nSVM RBF Evaluation:\n")
print(classification_report(y_test,y_pred, zero_division=0))

print("\n SVM RBF Confusion matrix :\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)

    
#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('RBF SVM Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')




plt.show()

#k nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

k_values=[i for i in range(10,31)]
scores=[]
error_rate=[]

for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(knn.score(X_test,y_test))
    error_rate.append(np.mean(y_pred!=y_test))


sns.lineplot(x=k_values, y=scores,marker='o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()


sns.lineplot(x=k_values, y=error_rate,marker='o')
plt.xlabel("K Values")
plt.ylabel("Error Rate")
plt.show()

#Get best k and train kNN

best_index=np.argmax(scores)
best_k=k_values[best_index]

knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("\nk = ",best_k)
print("\nkNN Evaluation:\n")
print(classification_report(y_test,y_pred))

print("\nkNN confusion matrix:\n")
cm=confusion_matrix(y_test,y_pred)
print(cm)
    
#Transform to df for easier plotting
cm_df = pd.DataFrame (cm,
                      index= ['0', '1'],
                      columns= ['0', '1'])

#Print the confusion matrix as heatmap
plt.figure(figsize=(5.5,4))
sns.heatmap (cm_df, annot=True)
plt.title('kNN Confusion Matrix \nAccuracy:{0:.3f}'.format\
          (accuracy_score (y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\resampled_datacombinationOver0.7Under1.csv")

# Split into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Convert categorical variable 'sex' to numerical format using one-hot encoding
categorical_features = ['sex', 'cp', 'restecg', 'exang', 'slope', 'smoke', 'ca']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


numeric_features = ['age', 'chol', 'cigs', 'trestbps', 'thaldur','thaltime','thalach','chol','thal', 'fbs','oldpeak']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define logistic regression model pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression())])

# Perform 10-fold cross-validation on the training data
for i in range(2,30):
    scores = cross_val_score(clf, X_train, y_train, cv=i,error_score='raise')
    # Print the mean accuracy score and standard deviation
    print("fold "+ str(i)+": Accuracy: %0.2f (+/- %0.2f)" %(scores.mean(), scores.std() * 2))


#SVM Accuracy
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels to integer values
X_train['sex'] = label_encoder.fit_transform(X_train['sex'])
y_train = label_encoder.fit_transform(y_train)


# Create an instance of SVM classifier
model = SVC(kernel='linear')

# Fit the SVM classifier to the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))


#Logisctic Regression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# Fit logistic regression model
clf.fit(X_train, y_train)

# Predict probabilities and generate ROC curve
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%% (+/- %0.2f)" % (accuracy * 100.0, accuracy.std() * 2))


# y_true is an array of true binary labels (0 or 1)
# y_score is an array of predicted probabilities of the positive class
x = roc_auc_score(y_test , y_prob)
print("Area under curve(AUC): " + str(x))


#Features Important Analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

clf.fit(X_train, y_train)

# Evaluate accuracy on testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy:", accuracy)

# Get feature importances
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
importances = result.importances_mean

# Sort feature importances in descending order
indices = importances.argsort()[::-1]

# Print feature importances
for i in indices:
    print(X.columns[i], ':', importances[i])


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

average_precision = average_precision_score(y_test, y_pred)
print('Average Precision:', average_precision)


clf = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(solver="liblinear"))])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



#Accuracy of sample, population accuracy
# Sample size
sample_size = 400

# Simple random sample
simple_sample = data.sample(sample_size)

pipeline = Pipeline([
    ('preprocessing', ColumnTransformer([
        ('numeric', StandardScaler(), ['age', 'chol', 'cigs', 'trestbps', 'thaldur','thaltime','thalach','chol','thal', 'fbs','oldpeak']),
        ('categorical', OneHotEncoder(), ['sex', 'cp', 'restecg', 'exang', 'slope', 'smoke', 'ca'])
    ])),
    ('kmeans', KMeans(n_clusters=3))
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

pipeline.fit(data)

sample_transformed = pipeline.transform(simple_sample)
test_transformed = pipeline.transform(X_test)


# Calculate the mean of column 'A' in the population
population_mean = data['age'].mean()
    
# Calculate the mean of column 'A' in the sample
sample_mean = simple_sample['age'].mean()

# Calculate the accuracy of the sample
accuracy = 100 * (1 - abs(population_mean - sample_mean) / population_mean)

print("\nPopulation mean of column 'Age':", population_mean)
print("Sample mean of column 'Age':", sample_mean)
print("Accuracy:", accuracy)

# Calculate the mean of column 'A' in the population
population_mean = data['chol'].mean()
    
# Calculate the mean of column 'A' in the sample
sample_mean = simple_sample['chol'].mean()

# Calculate the accuracy of the sample
accuracy = 100 * (1 - abs(population_mean - sample_mean) / population_mean)

print("\nPopulation mean of column 'Cholesterol':", population_mean)
print("Sample mean of column 'Cholesterol':", sample_mean)
print("Accuracy:", accuracy)

# Calculate the mean of column 'A' in the population
population_mean = data['restecg'].mean()
    
# Calculate the mean of column 'A' in the sample
sample_mean = simple_sample['restecg'].mean()

# Calculate the accuracy of the sample
accuracy = 100 * (1 - abs(population_mean - sample_mean) / population_mean)

print("\nPopulation mean of column 'RestingBP':", population_mean)
print("Sample mean of column 'RestingBP':", sample_mean)
print("Accuracy:", accuracy)

# Calculate the mean of column 'A' in the population
population_mean = data['fbs'].mean()
    
# Calculate the mean of column 'A' in the sample
sample_mean = simple_sample['fbs'].mean()

# Calculate the accuracy of the sample
accuracy = 100 * (1 - abs(population_mean - sample_mean) / population_mean)

print("\nPopulation mean of column 'FastingBS':", population_mean)
print("Sample mean of column 'FastingBS':", sample_mean)
print("Accuracy:", accuracy)


# Calculate the mean of column 'A' in the population
population_mean = data['oldpeak'].mean()
    
# Calculate the mean of column 'A' in the sample
sample_mean = simple_sample['oldpeak'].mean()

# Calculate the accuracy of the sample
accuracy = 100 * (1 - abs(population_mean - sample_mean) / population_mean)

print("\nPopulation mean of column 'Oldpeak':", population_mean)
print("Sample mean of column 'Oldpeak':", sample_mean)
print("Accuracy:", accuracy)



#oversampling accuracy
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Create a RandomOverSampler object
oversampler = RandomOverSampler(random_state=42)

# Use the object to oversample the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

clf = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression())])

# Train your model on the oversampled data
clf.fit(X_train_resampled, y_train_resampled)

# Use the model to make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(accuracy))


#Mean Imputation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

# load data into a pandas DataFrame
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\resampled_datacombinationOver0.7Under1.csv")


# separate target variable and features
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include='object').columns

# mean imputation for numeric columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())

# mode imputation for categorical columns
X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])


# create an instance of OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# fit the encoder on the training data
encoder.fit(X_train)

# transform the training and test data
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)


# train a decision tree classifier on the imputed data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_encoded, y_train)

# predict on test set and calculate accuracy
y_pred = clf.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of mean imputation: {accuracy:.2f}")



#KNN Imputation
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load the dataset
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\resampled_datacombinationOver0.7Under1.csv")


# split the data into training and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# preprocess the categorical variables
le = LabelEncoder()
for col in train.select_dtypes(include='object'):
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# perform K-nearest neighbor imputation
imputer = KNNImputer(n_neighbors=3)
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# convert the imputed arrays back to dataframes
train_imputed_df = pd.DataFrame(train_imputed, columns=train.columns)
test_imputed_df = pd.DataFrame(test_imputed, columns=test.columns)

# train a KNN classifier on the imputed data
k = 5
X_train = train_imputed_df.drop('HeartDisease', axis=1)
y_train = train_imputed_df['HeartDisease']
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# predict on the test set
X_test = test_imputed_df.drop('HeartDisease', axis=1)
y_test = test_imputed_df['HeartDisease']
y_pred = knn.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of KNN Imputation:", accuracy)


#Clustering imputation
import pandas as pd
import random

# Load the data into a pandas dataframe
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\resampled_datacombinationOver0.7Under1.csv")

# Identify the clusters based on a relevant variable, e.g., location
clusters = data.groupby('Age')

# Select a random sample of clusters
num_clusters = 10  # specify the number of clusters to select
selected_clusters = random.sample(list(clusters.groups.keys()), num_clusters)

# Create a list to store the sampled data
sampled_data = []

# Iterate over the selected clusters and include all individuals within each cluster
for cluster in selected_clusters:
    sampled_data += clusters.get_group(cluster).values.tolist()

# Convert the list of sampled data to a pandas dataframe
sampled_df = pd.DataFrame(sampled_data, columns=data.columns)

# Calculate the accuracy by comparing the sample mean to the population mean
population_mean = data['HeartDisease'].mean()
sample_mean = sampled_df['HeartDisease'].mean()
accuracy = (sample_mean / population_mean) 

#print(f'Sample mean: {sample_mean}')
#print(f'Population mean: {population_mean}')
print(f'Accuracy: {accuracy}')
