'''
learning.py
Learning malicious files and legitimate files
'''

import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree
import joblib
import sklearn.linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Loading the initial dataset delimited by '|'
data = pd.read_csv('data.csv', sep='|')
print(data.head())
print(data.describe())


# Dropping columns like Name of the file, MD5 (message digest) and label
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

print('Researching important feature based on %i total features\n' % X.shape[1])

# Feature selection using Trees Classifier
extratrees = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y ,test_size=0.2)

features = []
print('%i features identified as important:' % nb_features)
indices = np.argsort(extratrees.feature_importances_)[::-1][:nb_features]

for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], extratrees.feature_importances_[indices[f]]))

for f in sorted(np.argsort(extratrees.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

#Building the below Machine Learning model
model = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
        "RandomForest": ske.RandomForestClassifier(n_estimators=50),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
        "GNB": GaussianNB()
    }

#Training each of the model with the X_train and testing with X_test. The model with best accuracy will be ranked as winner
results = {}
print("\nNow testing model")

for algo in model:
    clf = model[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))

# Save the model and the feature list for later predictions
print('Saving algorithm and feature list in classifier directory...')
joblib.dump(model[winner], 'classifier/classifier.pkl')
with open('classifier/features.pkl', 'wb') as f:
    f.write(pickle.dumps(features))
print('Saved')

# Identify false and true positive rates
clf = model[winner]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
