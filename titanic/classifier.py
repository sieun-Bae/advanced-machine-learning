from transform import transform_features
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import numpy as np
import pandas as pd

titanic_df = pd.read_csv('./train.csv')
y_titanic_df = titanic_df['Survived'] #target
X_titanic_df = titanic_df.drop('Survived', axis = 1)

X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
dt_clf = DecisionTreeClassifier(random_state = 11)
rf_clf = RandomForestClassifier(random_state = 11)
lr_clf = LogisticRegression()

params = {'max_depth':[2,3,5,10], 'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

def train_all():
	dt_clf.fit(X_train, y_train)
	dt_pred = dt_clf.predict(X_test)
	print('\n')
	print('DecisionTreeClassifer 정확도:', accuracy_score(y_test, dt_pred))

	rf_clf.fit(X_train, y_train)
	rf_pred = rf_clf.predict(X_test)
	print('RandomForestClassifier 정확도:', accuracy_score(y_test, rf_pred))

	lr_clf.fit(X_train, y_train)
	lr_pred = lr_clf.predict(X_test)
	print('LogisticRegression 정확도:', accuracy_score(y_test, lr_pred))

def exec_kfold(clf, folds=5):
	kfold = KFold(n_splits=folds)
	scores = []
	print('\nKFold')
	for iter_count, (train_idx, test_idx) in enumerate(kfold.split(X_titanic_df)):
		X_train, X_test = X_titanic_df.values[train_idx], X_titanic_df.values[test_idx]
		y_train, y_test = y_titanic_df.values[train_idx], y_titanic_df.values[test_idx]

		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)

		scores.append(accuracy)
		print('교차검증 {0} 정확도: {1}'.format(iter_count, accuracy))

def cross_val(clf, folds=5):
	scores = cross_val_score(clf, X_titanic_df, y_titanic_df, cv=folds)
	print('\nCross Validation')
	for iter_count, accuracy in enumerate(scores):
		print('교차검증 {0} 정확도: {1}'.format(iter_count, accuracy))
	print('평균 정확도:', np.mean(scores))

def grid_search(clf, params=params, folds=5):
	print('\nGridSearchCV')
	grid_dclf = GridSearchCV(clf, param_grid = params, scoring = 'accuracy', cv=folds)
	grid_dclf.fit(X_train, y_train)

	print('최적 하이퍼파라미터:', grid_dclf.best_params_)
	print('최고 정확도:', grid_dclf.best_score_)
	best_dclf = grid_dclf.best_estimator_

	dpredictions = best_dclf.predict(X_test)
	accuracy = accuracy_score(y_test, dpredictions)

	print('최고 테스트 정확도:', accuracy)

if __name__ == '__main__':
	exec_kfold(dt_clf, folds = 5)
	'''
	KFold
	교차검증 0 정확도: 0.7541899441340782
	교차검증 1 정확도: 0.7808988764044944
	교차검증 2 정확도: 0.7865168539325843
	교차검증 3 정확도: 0.7696629213483146
	교차검증 4 정확도: 0.8202247191011236
	'''
	cross_val(dt_clf, folds = 5)
	'''
	Cross Validation == stratified kfold
	교차검증 0 정확도: 0.7430167597765364
	교차검증 1 정확도: 0.7752808988764045
	교차검증 2 정확도: 0.7921348314606742
	교차검증 3 정확도: 0.7865168539325843
	교차검증 4 정확도: 0.8426966292134831
	평균 정확도: 0.7879291946519366
	'''
	grid_search(dt_clf, folds = 5)
	'''
	최적 하이퍼파라미터: {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
	최고 정확도: 0.7991825076332119
	최고 테스트 정확도: 0.8715083798882681
	'''
	train_all()
	'''
	DecisionTreeClassifer 정확도: 0.7877094972067039
	RandomForestClassifier 정확도: 0.8547486033519553
	LogisticRegression 정확도: 0.8491620111731844
	'''











