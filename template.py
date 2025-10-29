# Template parameters for Logistic Regression Models

from sklearn.linear_model import LogisticRegression

logistic_pipeline = Pipeline ([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())]
)

parameters_logistics = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__max_iter': [100, 200, 500]
}

logistic_model = GridSearchCV(logistic_pipeline, parameters_logistics, cv=5, scoring = 'accuracy')
logistic_model.fit(x_train, y_train)


# Template parameters for kNeighborsClassifier Models

from sklearn.neighbors import KNeighborsClassifier
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

parameters_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

knn_model = GridSearchCV(knn_pipeline, parameters_knn, cv=5, scoring='accuracy')
knn_model.fit(x_train, y_train)



# Template parameters for Support Vector Models

from sklearn.svm import SVC
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

parameters_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto']
}

svm_model = GridSearchCV(svm_pipeline, parameters_svm, cv=5, scoring = 'accuracy')
svm_model.fit(x_train, y_train)



# Template parameters for DecisionTreeClassififer Models

from sklearn.tree import DecisionTreeClassifier
decision_tree_pipeline = Pipeline ([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier())
])

parameters_decisiontree = {
     'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

decision_tree_model = GridSearchCV(decision_tree_pipeline, parameters_decisiontree, cv=5, scoring = 'accuracy')

decision_tree_model.fit(x_train, y_train)



# Template parameters for Random ForestClassifier Models

from sklearn.ensemble import RandomForestClassifier
randomclassifier_pipeline = Pipeline ([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

parameters_randomclassifier = {
}

random_classifier_model = GridSearchCV(RandomForestClassifier(), parameters_randomclassifier, cv=5, scoring = 'accuracy')
random_classifier_model.fit(x_train, y_train)