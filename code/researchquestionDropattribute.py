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



