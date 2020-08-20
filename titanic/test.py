from transform import *
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv('./train.csv')
y_titanic_df = titanic_df['Survived'] #target
X_titanic_df = titanic_df.drop('Survived', axis = 1)

X_titanic_df = transform_features(X_titanic_df)

def main():
	X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
	dt_clf = DecisionTreeClassifer()
	rf_clf = RandomForestClassifier()
	lr_clf = LogisticRegression()

	dt_clf.fit(X_train, y_train)
	dt_pred - dt_clf.predict(X_test)
	print('DecisionTreeClassifer 정확도:', accuracy_score(y_test, dt_pred))

	rf_clf.fit(X_train, y_train)
	rf_pred = rf_clf.predict(X_test)
	print('RandomForestClassifier 정확도:', accuracy_score(y_test, rf_pred))

	lr_clf.fit(X_train, y_train)
	lr_prdd = lr_clf.predict(X_test)
	print('LogisticRegression 정확도:', accuracy_score(y_test, lr_pred))

if __name__ == '__main__':
	main()