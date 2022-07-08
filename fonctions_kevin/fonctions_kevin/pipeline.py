import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer


from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer





############################ Pipeline

# Variables du même type

# Always scale the input. The most convenient way is to use a pipeline.
# Si uniquement des variables numériques ici

clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))

# clf.fit(X_train, y_train)

# ---------------------------------------------------

# Variables de type différent et 1 seul traitement

numerical_features = ['sepallength', 'sepalwidth']
categorical_features = [...]

transformer = make_column_transformer((StandardScaler(), numerical_features), # numerique
                                      (SimpleImputer(strategy= 'most_frequent'), categorical_features)) # texte

clf = make_pipeline(transformer,
                    SGDClassifier(max_iter=1000, tol=1e-3))

# clf.fit(X_train, y_train)


# -------------------------------------------------

#  Variable de type différent et plusieurs traitements

numerical_features = ['sepallength', 'sepalwidth']
categorical_features = [...]


numerical_pipeline = make_pipeline(SimpleImputer(),
                                   StandardScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                                   OneHotEncoder())


preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                        (categorical_pipeline, categorical_features))


clf = make_pipeline(preprocessor, SGDClassifier())

# clf.fit(X_train, y_train)

# ----------------------------------------------------

# Prendre toutes les variables de type différent et plusieurs traitements



numerical_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)


numerical_pipeline = make_pipeline(SimpleImputer(),
                                   StandardScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                                   OneHotEncoder())


preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                        (categorical_pipeline, categorical_features))


clf = make_pipeline(preprocessor, SGDClassifier())

# clf.fit(X_train, y_train)


# -----------------------------------------------------

# et après :

# clf.score(X_test, y_test)
# cross_validation = cross_val_score(clf, X_test, y_test, cv=5).mean()
# clf.predict(iris_pipeline.drop(['class'], axis=1))