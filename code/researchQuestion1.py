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



