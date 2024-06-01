import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

data = pd.read_csv("placedata v2.0 synthetic.csv")
print(list(data.columns))
data.head()

data.tail()

lb = LabelEncoder()
data['PlacementStatus'] = lb.fit_transform(data['PlacementStatus'])
data['PlacementTraining'] = lb.fit_transform(data['PlacementTraining'])
data['ExtracurricularActivities'] = lb.fit_transform(data['ExtracurricularActivities'])
data = data.drop(['StudentID'], axis = 1)
outputs = data['PlacementStatus']
inputs = data.drop(['PlacementStatus'], axis = 1)
std = StandardScaler()
inputs = std.fit_transform(inputs)

data.isna().sum()

data.describe()

plt.figure(figsize = (15,10))
sns.heatmap(data.corr(),linewidths = 3, annot=True)
plt.show()

# Visualizating effect of internships on placement
Internships_effect = data.pivot_table(index = 'PlacementStatus',values="CGPA", columns='Internships', aggfunc='count')
Internships_effect

X = Internships_effect.columns
X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Internships_effect.iloc[0,:], 0.4, label = 'NotPlaced')
plt.bar(X_axis + 0.2, Internships_effect.iloc[1,:], 0.4, label = 'Placed')

plt.xticks(X_axis, X)
plt.xlabel("Number of internships")
plt.ylabel("Number of placements")
plt.title("internships effect")
plt.legend()
plt.show()

# Effect of CGPA and softskills
CGPA_effect = data.pivot_table(index = 'PlacementStatus', values= ['CGPA', 'SoftSkillsRating'])
CGPA_effect
# Shows that higher cgpa and softskillsscore are more probable to land student a placement

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2, random_state = 0)
models = [[SVC(), "Support vector"],
        [LogisticRegression(C=10), "Logistic regression"],
         [RandomForestClassifier(n_estimators = 20), "Random Forest"],
         [DecisionTreeClassifier(max_depth = 7), "Decision Trees"],
         [KNeighborsClassifier(n_neighbors = 7), "KNeighbourClassifier"],
         [xgb.XGBClassifier(objective="binary:logistic", random_state=42), "XGBoost"],
         [AdaBoostClassifier(n_estimators = 25), "Adaboost"],
         [GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=0), "Gradient Boosting"]]
for i in models:
    name = i[1]
    model = i[0]
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred) * 100)
    cnf = confusion_matrix(y_test,y_pred)
    fig, ax = plot_confusion_matrix(conf_mat = cnf)
    plt.show()
    print("\n")

model = AdaBoostClassifier()
grid = dict()
grid['n_estimators'] = [85,89,90,91,95]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
grid_result = grid_search.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

y_pred = grid_result.predict(X_test)
print(accuracy_score(y_test, y_pred) * 100)
cnf = confusion_matrix(y_test,y_pred)
fig, ax = plot_confusion_matrix(conf_mat = cnf)
plt.show()

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
hello = best_model.predict(X_test)
print('the accuracy is : ',accuracy_score(y_test, hello)*100)
joblib.dump(best_model,'Placement.joblib')