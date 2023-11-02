from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np


class RandomForest():
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = [None, 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        print(random_grid)
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(x_train, y_train)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_
        self.model = best_random
        # self.train(x_train, y_train)
        # print(f"Train score: {self.model.score(x_train, y_train)}")
        # print(f"Test score: {self.model.score(x_test, y_test)}")


class GradientBoosting():
        def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
            self.model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
    
        def train(self, x_train, y_train):
            self.model.fit(x_train, y_train)
    
        def predict(self, x_test):
            return self.model.predict(x_test)
    
        def predict_proba(self, x_test):
            return self.model.predict_proba(x_test)
        
        def search_best_params(self, x_train, y_train, x_test, y_test):
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'bootstrap': bootstrap}
            print(random_grid)
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            rf = GradientBoostingClassifier()
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                            n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
            # Fit the random search model
            rf_random.fit(x_train, y_train)
            print(rf_random.best_params_)
            best_random = rf_random.best_estimator_
            self.model = best_random


class GaussianNaiveBayes():
    def __init__(self, priors=None, var_smoothing=1e-09):
        self.model = GaussianNB(priors=priors, var_smoothing=var_smoothing)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        class_priors = [None, [0.3, 0.7], [0.1, 0.9], [0.01, 0.99]]
        var_smoothing = [1e-09, 1e-08, 1e-07]
        random_grid = {'priors': class_priors,
                        'var_smoothing': var_smoothing}
        print(random_grid)
        gnb = GaussianNB()
        rf_random = RandomizedSearchCV(estimator=gnb, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        
        rf_random.fit(x_train, y_train)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_
        self.model = best_random


class DecisionTree():
    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, class_weight=None, random_state=None, ccp_alpha=0.0, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0):
        self.model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, class_weight=class_weight, random_state=random_state, ccp_alpha=ccp_alpha, min_impurity_decrease=min_impurity_decrease, min_weight_fraction_leaf=min_weight_fraction_leaf)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        # Number of trees in random forest
        criterion = ['gini', 'entropy']
        splitter = ['best', 'random']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        max_features = [None, 'sqrt']
        class_weight = [None, 'balanced']
        presort = [True, False]
        ccp_alpha = [0.0, 0.1, 0.2, 0.3]
        min_impurity_decrease = [0.0, 0.1, 0.2, 0.3]
        min_weight_fraction_leaf = [0.0, 0.1, 0.2, 0.3]
        random_grid = {'criterion': criterion,
                        'splitter': splitter,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features,
                        'class_weight': class_weight,
                        'presort': presort,
                        'ccp_alpha': ccp_alpha,
                        'min_impurity_decrease': min_impurity_decrease,
                        'min_weight_fraction_leaf': min_weight_fraction_leaf
                        }
        print(random_grid)
        dt = DecisionTreeClassifier()
        rf_random = RandomizedSearchCV(estimator=dt, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        
        rf_random.fit(x_train, y_train)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_
        self.model = best_random


class LogisticRegression():
    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, class_weight=None):
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio, class_weight=class_weight)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        # Create regularization penalty space
        penalty = ['l1', 'l2']
        # Create regularization hyperparameter space
        C = np.logspace(0, 4, 10)
        # Create hyperparameter options
        hyperparameters = dict(C=C, penalty=penalty)
        # Create grid search using 5-fold cross validation
        clf = GridSearchCV(self.model, hyperparameters, cv=5, verbose=0)
        # Fit grid search
        best_model = clf.fit(x_train, y_train)
        print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
        print('Best C:', best_model.best_estimator_.get_params()['C'])
        self.model = best_model.best_estimator_
        # self.train(x_train, y_train)
        # print(f"Train score: {self.model.score(x_train, y_train)}")
        # print(f"Test score: {self.model.score(x_test, y_test)}")


class Perceptron():
    def __init__(self, penality=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False):
            self.model = Perceptron(penality=penality, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, eta0=eta0, n_jobs=n_jobs, random_state=random_state, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = [None, 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        print(random_grid)
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(x_train, y_train)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_
        self.model = best_random


class MLP():
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000, class_weight=None):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun, class_weight=class_weight)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def search_best_params(self, x_train, y_train, x_test, y_test):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = [None, 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        print(random_grid)
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(x_train, y_train)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_
        self.model = best_random
