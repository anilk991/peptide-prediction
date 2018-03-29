import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score




pos = pd.read_csv("E:/WorkingDir/data_scripts/pos.fasta", header = None)
neg = pd.read_csv("E:/WorkingDir/data_scripts/neg.fasta", header = None)
pos.columns =  ['Sequences']
neg.columns =  ['Sequences']
pos = pos.drop(pos[pos.Sequences.str.contains('^>')].index)
neg = neg.drop(neg[neg.Sequences.str.contains('^>')].index)

pos['Label'] = 1
neg['Label'] = -1

#pos['Label'] = 'Positive'
#neg['Label'] = 'Negative'
frames = [pos, neg]
df = pd.concat(frames, ignore_index = True)
#df_tf = df.Sequences.str.contains('^>')
#df = df.drop(df[df.Sequences.str.contains('^>')].index)
#df = shuffle(df)

Z = df.Sequences
y = df.Label

#le = LabelEncoder()
#le.fit(y)

X_1=[]
cols = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']




l = []
for row in Z:
	l=((row.count('A')/len(row)), (row.count('C')/len(row)), (row.count('D')/len(row)),
		(row.count('E')/len(row)), (row.count('F')/len(row)), (row.count('G')/len(row)),
		(row.count('H')/len(row)), (row.count('I')/len(row)), (row.count('K')/len(row)),
		(row.count('L')/len(row)), (row.count('M')/len(row)), (row.count('N')/len(row)),
		(row.count('P')/len(row)), (row.count('Q')/len(row)), (row.count('R')/len(row)),
		(row.count('S')/len(row)), (row.count('T')/len(row)), (row.count('V')/len(row)),
		(row.count('W')/len(row)), (row.count('Y')/len(row)))
	X_1.append(l)
#print(len(X_1))

X = pd.DataFrame(X_1, columns = cols)
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.fit_transform(X_test)

from sklearn.model_selection import GridSearchCV
'''
#parameters for SVC were found using gridsearchCV
pipe1 = Pipeline([('sc', StandardScaler()), ('clf', SVC(random_state=0, probability=True, C=1.0, gamma=0.1, kernel='rbf'))])

#param_range = [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
#param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
#gs = GridSearchCV(estimator=pipe1, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
#scores = []
#scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
#print("CV accuracy: %.3f +/- %.3f" % np.mean(scores), np.std(scores))


#gs.fit(X_train, y_train)
#print(gs.best_score_)
#print(gs.best_params_)


pipe1.fit(X_train_std, y_train)



scores_1 = cross_val_score(estimator=pipe1, X=X_train, y=y_train, cv=10, n_jobs=1)
print("CV Accuracy Scores: %s" % scores_1)
print("CV Accuracy: %.3f +/- %.3f" % (np.mean(scores_1), np.std(scores_1)))
pipe1_test = pipe1.score(X_test_std, y_test)
print("Pipe1 Accuracy: ", pipe1_test)

'''
from sklearn.ensemble import RandomForestClassifier
#parameters criterion, max_features and n_estimators in pipe2 were found using gridsearchCV
pipe2 = Pipeline([('sc', StandardScaler()), ('clf2', RandomForestClassifier(criterion='entropy', max_features='log2', n_estimators=1000, n_jobs=2))])

#param_range2 = [1, 10, 100, 1000, 10000]
#param_grid = [{'clf2__n_estimators': param_range2, 'clf2__max_features': ['sqrt'], 'clf2__criterion': ['gini']}, {'clf2__n_estimators': param_range2, 'clf2__max_features': ['sqrt'], 'clf2__criterion': ['entropy']}, {'clf2__n_estimators': param_range2, 'clf2__max_features': ['log2'], 'clf2__criterion': ['gini']}, {'clf2__n_estimators': param_range2, 'clf2__max_features': ['log2'], 'clf2__criterion': ['entropy']}]
#gs = GridSearchCV(estimator=pipe2, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
#gs.fit(X_train, y_train)
#print(gs.best_score_)
#print(gs.best_params_)

pipe2.fit(X_train, y_train)
scores_2 = cross_val_score(estimator=pipe2, X=X_train, y=y_train, cv=10, n_jobs=1)
print("CV Accuracy Scores: %s" % scores_1)
print("CV Accuracy: %.3f +/- %.3f" % (np.mean(scores_2), np.std(scores_2)))
pipe2_test = pipe2.score(X_test_std, y_test)
print("Pipe1 Accuracy: ", pipe2_test)
