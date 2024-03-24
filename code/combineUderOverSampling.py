

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







