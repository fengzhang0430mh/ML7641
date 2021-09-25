#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


loans = pd.read_csv('loan_data.csv')

loans.info()
loans.head()


cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=2)
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

### neural network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
predictions_nn = clf.predict(X_test)
print(classification_report(y_test,predictions_nn))
print(confusion_matrix(y_test,predictions_nn))


### Boosting
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions_boosting = rfc.predict(X_test)
print(classification_report(y_test,predictions_boosting))
print(confusion_matrix(y_test,predictions_boosting))


### support vector machine
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train,y_train)
predictions_svc = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions_svc))
print(classification_report(y_test,predictions_svc))



model = SVC(kernel="poly")
model.fit(X_train,y_train)
predictions_svc = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions_svc))
print(classification_report(y_test,predictions_svc))


## knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

### report

## Tree
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
outcome = []
train_outcome = []
for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = dtree.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


dtree = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=3)
outcome = []
train_outcome = []
for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = dtree.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)



### boosting

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    rfc.fit(X_train,y_train)
    predictions = rfc.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = rfc.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


### svm

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = model.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


### kNn

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = knn.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


## NN

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = clf.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)



### second data
train = pd.read_csv('titanic_train.csv')
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop('Cabin',axis=1,inplace=True)
train.drop(['Sex','Embarked','Name','Ticket', "PassengerId"],axis=1,inplace=True)
# train = pd.concat([train,sex,embark],axis=1)
train = train.dropna(axis = 0, how ='any')  
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


### for the report 
## Tree
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
outcome = []
train_outcome = []
for i in x:

    X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=i, random_state=101)

    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = dtree.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


dtree = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=3)
outcome = []
train_outcome = []
for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = dtree.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)



### boosting

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    rfc.fit(X_train,y_train)
    predictions = rfc.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = rfc.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


### svm

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = model.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


### kNn

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = knn.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)


## NN

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = clf.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)
