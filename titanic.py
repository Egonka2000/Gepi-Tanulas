import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

def class_balance(titanic_df):
  survived_plot(titanic_df)
  survived = titanic_df[titanic_df['survived'] == 0].head(len(titanic_df[titanic_df['survived'] == 1]))
  not_survived = titanic_df[titanic_df['survived'] == 1]
  titanic_df = pd.concat([not_survived, survived])
  survived_plot(titanic_df)
  return titanic_df

def survived_plot(titanic_df):
  sns.countplot(data=titanic_df, x='survived', hue='survived')
  plt.show()

def roc_plot(y_test, y_pred_probs, classifier):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(classifier))
    plt.show()
    print(roc_auc_score(y_test, y_pred_probs))
  
titanic_df = pd.read_csv('Datasets/titanic.csv')

sns.heatmap(titanic_df.corr(), annot=True)
plt.show()

titanic_df = titanic_df.drop("embark_town", axis=1)
titanic_df = class_balance(titanic_df)

titanic_dummies = pd.get_dummies(titanic_df).astype('float32')

X = titanic_dummies.drop("survived", axis=1).values

y = titanic_dummies["survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

logreg_steps = [('scaler', StandardScaler()),
         ('logreg', LogisticRegression())]

knn_steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

tree_steps = [('scaler', StandardScaler()),
         ('tree', DecisionTreeClassifier())]

perc_steps = [('scaler', StandardScaler()),
         ('perc', Perceptron())]

gaus_steps = [('scaler', StandardScaler()),
         ('gaus', GaussianNB())]

logreg_pipeline = Pipeline(logreg_steps)
knn_pipeline = Pipeline(knn_steps)
tree_pipeline = Pipeline(tree_steps)
perc_pipeline = Pipeline(perc_steps)
gaus_pipeline = Pipeline(gaus_steps)

logreg_params = {"logreg__penalty": ["l1", "l2"],
         "logreg__tol": np.linspace(0.0001, 1.0, 50),
         "logreg__C": np.linspace(0.1, 1.0, 50),
         "logreg__class_weight": ["balanced", {0:0.8, 1:0.2}],
         "logreg__solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
         "logreg__multi_class": ['auto', 'ovr', 'multinomial']}

knn_params = {"knn__n_neighbors": np.arange(1, 20),
              "knn__weights": ['uniform', 'distance'],
              "knn__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
              "knn__leaf_size": np.arange(10, 100)}

tree_params = {"tree__criterion": ['gini', 'entropy', 'log_loss'],
               "tree__splitter": ['best', 'random'],
               "tree__max_features": ['auto', 'sqrt', 'log2'],
               "tree__ccp_alpha": np.linspace(0.0001, 1.0, 50)}

perc_params = {"perc__penalty": ['l2', 'l1', 'elasticnet'],
         "perc__tol": np.linspace(0.0001, 1.0, 50),
         "perc__alpha": np.linspace(0.0001, 1.0, 50),
         "perc__class_weight": ["balanced", {0:0.8, 1:0.2}],}

gaus_params = {"gaus__var_smoothing": np.linspace(0.0001, 1.0, 50)}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

logreg_cv = RandomizedSearchCV(logreg_pipeline, param_distributions=logreg_params, cv=kf, n_iter=20)
knn_cv = RandomizedSearchCV(knn_pipeline, param_distributions=knn_params, cv=kf, n_iter=20)
tree_cv = RandomizedSearchCV(tree_pipeline, param_distributions=tree_params, cv=kf, n_iter=20)
perc_cv = RandomizedSearchCV(perc_pipeline, param_distributions=perc_params, cv=kf, n_iter=20)
gaus_cv = RandomizedSearchCV(gaus_pipeline, param_distributions=gaus_params, cv=kf, n_iter=20) 

logreg_cv.fit(X_train, y_train)
knn_cv.fit(X_train, y_train)
tree_cv.fit(X_train, y_train)
perc_cv.fit(X_train, y_train)
gaus_cv.fit(X_train, y_train)

logreg_y_pred = logreg_cv.predict(X_test)
logreg_y_pred_probs = logreg_cv.predict_proba(X_test)[:, 1]

knn_y_pred = knn_cv.predict(X_test)
knn_y_pred_probs = knn_cv.predict_proba(X_test)[:, 1]

tree_y_pred = tree_cv.predict(X_test)
tree_y_pred_probs = tree_cv.predict_proba(X_test)[:, 1]

perc_y_pred = perc_cv.predict(X_test)
#perc_y_pred_probs = perc_cv.predict_proba(X_test)[:, 1]

gaus_y_pred = gaus_cv.predict(X_test)
gaus_y_pred_probs = gaus_cv.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, logreg_y_pred))
print(classification_report(y_test, logreg_y_pred))
roc_plot(y_test, logreg_y_pred_probs, "Logistic Regression")

print(confusion_matrix(y_test, knn_y_pred))
print(classification_report(y_test, knn_y_pred))
roc_plot(y_test, knn_y_pred_probs, "K Neighbors")

print(confusion_matrix(y_test, tree_y_pred))
print(classification_report(y_test, tree_y_pred))
roc_plot(y_test, tree_y_pred_probs, "Decision Tree")

print(confusion_matrix(y_test, perc_y_pred))
print(classification_report(y_test, perc_y_pred))
#roc_plot(y_test, perc_y_pred_probs, "Perception")

print(confusion_matrix(y_test, gaus_y_pred))
print(classification_report(y_test, gaus_y_pred))
roc_plot(y_test, gaus_y_pred_probs, "Naiv Bayes")