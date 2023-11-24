import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from gensim.models import Word2Vec
import seaborn as sns


def load_data_precise():
    data = pd.read_csv('../database/data/high_level_precise.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # mettre payment_method en majuscule
    data['payment_method'] = data['payment_method'].str.upper()

    # erreur dans les données, il faut checker si la méthode de paiement est VISA,MASTERCARD,AMEX et que le type de paiement est DATRS_RED alors il faut changer DATRS_RED en DATRS_HID, pareil pour ADYEN_JSHD et ADYEN_RED
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    return data


def load_new_data():
    data = pd.read_csv('../database/data/HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right_PAID.csv')
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])
    data.set_index('DATETIME', inplace=True)
    data['PAYMENT_METHOD'] = data['PAYMENT_METHOD'].str.upper()

    # montre le nombre de données qui ont la colonne ISSUER

    return data


def train_n_predict(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # print the data were the model predicted a wrong label
    # print(x_test[y_test != y_pred])

    conf_mat = confusion_matrix(y_test, y_pred)

    # print the confusion matrix
    print(conf_mat)
    # plot the confusion matrix, red for the errors, green for the correct predictions
    fig, ax = plt.subplots()
    ax.matshow(conf_mat, cmap=plt.cm.Reds, alpha=0.3)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')

    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.show()



def isolation_forest_new_data():
    data = pd.read_csv('../database/data/high_level_precise.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # mettre payment_method en majuscule
    data['payment_method'] = data['payment_method'].str.upper()

    # erreur dans les données, il faut checker si la méthode de paiement est VISA,MASTERCARD,AMEX et que le type de paiement est DATRS_RED alors il faut changer DATRS_RED en DATRS_HID, pareil pour ADYEN_JSHD et ADYEN_RED
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    
    # print the data from the 19th september 2023
    ogone = data.loc['2023-09-19']
    ogone = ogone[ogone['payment_type'] == 'OGONE_HID']
    # keep only data between 12:00 and 16:00
    ogone = ogone.between_time('12:00', '16:00')
    print(ogone)
     
    # make a boxplot of the paid rate, set the x axis vertical
    # data.boxplot(column=['paid_rate'], by=['payment_method'], vert=False, figsize=(20, 10))
    # plt.show()

    one_hot = OneHotEncoder()
    payment_types = data['payment_type'].values.reshape(-1, 1)
    payment_types_one_hot = one_hot.fit_transform(payment_types).toarray()
    payment_types_one_hot = pd.DataFrame(payment_types_one_hot, columns=one_hot.get_feature_names_out(), index=data.index)
    data = pd.concat([data, payment_types_one_hot], axis=1)

    # faire la meme chose mais pour 'payment_method'
    one_hot2 = OneHotEncoder()
    payment_methods = data['payment_method'].values.reshape(-1, 1)
    payment_methods_one_hot = one_hot2.fit_transform(payment_methods).toarray()
    payment_methods_one_hot = pd.DataFrame(payment_methods_one_hot, columns=one_hot2.get_feature_names_out(), index=data.index)
    data = pd.concat([data, payment_methods_one_hot], axis=1)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # print(train_data.head())
    # print(test_data.head())
    
    columns_to_train = train_data.columns.values.tolist()
    # remove the columns 'payment_method' and 'payment_type' from the columns to train
    columns_to_train.remove('payment_method')
    columns_to_train.remove('payment_type')
    print(columns_to_train)
    
    # input()

    random_state = np.random.RandomState(42)
    model = IsolationForest(n_estimators=100, max_samples="auto", contamination=float(0.1), random_state=random_state)
    # fit data without the columns 'payment_method' and 'payment_type'
    model.fit(train_data[columns_to_train])

    test_data['scores'] = model.decision_function(test_data[columns_to_train])
    test_data['anomaly'] = model.predict(test_data[columns_to_train])

    # print(test_data[['scores', 'anomaly']])

    # plt.scatter(test_data['total_transaction_count'], test_data['paid_rate'], c=test_data['anomaly'], cmap='viridis') # viridis
    # plt.xlabel('paid_rate')
    # plt.ylabel('total_transaction_count')
    # plt.colorbar(label='Outlier(1) / Normal(0)')
    # plt.show()
    

    # compute the number of outliers and normal points
    n_outliers = len(test_data[test_data['anomaly'] == -1])
    n_normals = len(test_data[test_data['anomaly'] == 1])
    print(f"Number of outliers: {n_outliers} / {len(test_data)}")
    print(f"Number of normal points: {n_normals} / {len(test_data)}")

    # print(test_data.head())

    september_2023 = test_data.loc['2023-09-01':'2023-09-30']
    # keep only the OGONE_HID data
    september_2023 = september_2023[september_2023['payment_type'] == 'OGONE_HID']

    print(f"Nb anomaly in september: {len(september_2023[september_2023['anomaly'] == -1])} / {len(september_2023)}")

    # print(september_2023[september_2023['anomaly'] == -1])
    anomalies_september = september_2023[september_2023['anomaly'] == -1]
    anomalies_september.sort_index(inplace=True)
    print(anomalies_september)

def isolation_forest_right_percent():
    data = pd.read_csv('../database/data/HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right_PAID.csv')
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])
    data.set_index('DATETIME', inplace=True)

    # mettre payment_method en majuscule
    data['PAYMENT_METHOD'] = data['PAYMENT_METHOD'].str.upper()

    # drop column 'STATE'
    data.drop(columns=['STATE'], inplace=True)

    # remove the entry where data['PAYMENT_METHOD'] == 'UNKNOWN' or data['PAYMENT_TYPE'] == 'UNKNOWN'
    data = data[(data['PAYMENT_METHOD'] != 'UNKNOWN') & (data['PAYMENT_TYPE'] != 'UNKNOWN')]
    
    # print the data from the 19th september 2023
    ogone = data.loc['2023-09-19']
    ogone = ogone[ogone['PAYMENT_TYPE'] == 'OGONE_HID']
    # keep only data between 12:00 and 16:00
    # ogone = ogone.between_time('12:00', '16:00')
    # print(ogone)

     
    # make a boxplot of the paid rate, set the x axis vertical
    # data.boxplot(column=['paid_rate'], by=['payment_method'], vert=False, figsize=(20, 10))
    # plt.show()

    # one_hot = OneHotEncoder()
    # payment_types = data['PAYMENT_TYPE'].values.reshape(-1, 1)
    # payment_types_one_hot = one_hot.fit_transform(payment_types).toarray()
    # payment_types_one_hot = pd.DataFrame(payment_types_one_hot, columns=one_hot.get_feature_names_out(), index=data.index)
    # data = pd.concat([data, payment_types_one_hot], axis=1)

    # # faire la meme chose mais pour 'payment_method'
    # one_hot2 = OneHotEncoder()
    # payment_methods = data['PAYMENT_METHOD'].values.reshape(-1, 1)
    # payment_methods_one_hot = one_hot2.fit_transform(payment_methods).toarray()
    # payment_methods_one_hot = pd.DataFrame(payment_methods_one_hot, columns=one_hot2.get_feature_names_out(), index=data.index)
    # data = pd.concat([data, payment_methods_one_hot], axis=1)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # print(train_data.head())
    # print(test_data.head())
    
    columns_to_train = train_data.columns.values.tolist()
    # remove the columns 'payment_method' and 'payment_type' from the columns to train
    columns_to_train.remove('PAYMENT_METHOD')
    columns_to_train.remove('PAYMENT_TYPE')
    # columns_to_train.remove('REAL_TRANSACTION_COUNT')
    print(columns_to_train)
    
    # print(train_data[columns_to_train].head())
    # input()

    random_state = np.random.RandomState(34)
    model = IsolationForest(n_estimators=100, max_samples="auto", contamination=float(0.05), random_state=random_state)
    # fit data without the columns 'payment_method' and 'payment_type'
    model.fit(train_data[columns_to_train])

    test_data['scores'] = model.decision_function(test_data[columns_to_train])
    test_data['anomaly'] = model.predict(test_data[columns_to_train])

    # print(test_data[['scores', 'anomaly']])

    # plt.scatter(test_data['REAL_TRANSACTION_COUNT'], test_data['REAL_PERCENTAGE'], c=test_data['anomaly'], cmap='viridis') # viridis
    # plt.xlabel('paid_rate')
    # plt.ylabel('total_transaction_count')
    # plt.colorbar(label='Outlier(1) / Normal(0)')
    # plt.show()
    

    # compute the number of outliers and normal points
    n_outliers = len(test_data[test_data['anomaly'] == -1])
    n_normals = len(test_data[test_data['anomaly'] == 1])
    print(f"Number of outliers: {n_outliers} / {len(test_data)}")
    print(f"Number of normal points: {n_normals} / {len(test_data)}")

    # print(test_data.head())

    september_2023 = test_data.loc['2023-09-01':'2023-09-30']
    # keep only the OGONE_HID data
    september_2023 = september_2023[september_2023['PAYMENT_TYPE'] == 'OGONE_HID']

    print(f"Nb anomaly in september: {len(september_2023[september_2023['anomaly'] == -1])} / {len(september_2023)}")

    # print(september_2023[september_2023['anomaly'] == -1])
    anomalies_september = september_2023[september_2023['anomaly'] == -1]
    anomalies_september.sort_index(inplace=True)
    print(anomalies_september)

    # print the 19th september 2023
    print(september_2023.loc['2023-09-19'][september_2023['anomaly'] == -1])


# isolation_forest_new_data()

def heatmaps():
    data = pd.read_csv('../database/data/high_level_precise.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # mettre payment_method en majuscule
    data['payment_method'] = data['payment_method'].str.upper()

    # erreur dans les données, il faut checker si la méthode de paiement est VISA,MASTERCARD,AMEX et que le type de paiement est DATRS_RED alors il faut changer DATRS_RED en DATRS_HID, pareil pour ADYEN_JSHD et ADYEN_RED
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
    data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
    data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'

    # data = load_new_data()

    # take only the 19th september 2023
    data = data.loc['2023-09-19']

    one_hot = OneHotEncoder()
    payment_types = data['payment_type'].values.reshape(-1, 1)
    payment_types_one_hot = one_hot.fit_transform(payment_types).toarray()
    payment_types_one_hot = pd.DataFrame(payment_types_one_hot, columns=one_hot.get_feature_names_out(), index=data.index)
    data = pd.concat([data, payment_types_one_hot], axis=1)

    one_hot2 = OneHotEncoder()
    payment_methods = data['payment_method'].values.reshape(-1, 1)
    payment_methods_one_hot = one_hot2.fit_transform(payment_methods).toarray()
    payment_methods_one_hot = pd.DataFrame(payment_methods_one_hot, columns=one_hot2.get_feature_names_out(), index=data.index)
    data = pd.concat([data, payment_methods_one_hot], axis=1)

    hours = data.index.hour
    attributes = data.columns.values.tolist()
    attributes.remove('payment_method')
    attributes.remove('payment_type')
    attributes.remove('paid_rate')
    attributes.remove('total_transaction_count')

    matrix_data_mean = pd.DataFrame(index=hours, columns=attributes)
    matrix_data_std = pd.DataFrame(index=hours, columns=attributes)
    matrix_data_sum = pd.DataFrame(index=hours, columns=attributes)

    for attribute in attributes:
        matrix_data_mean[attribute] = data[attribute].groupby(hours).mean()
        matrix_data_std[attribute] = data[attribute].groupby(hours).std()
        matrix_data_sum[attribute] = data[attribute].groupby(hours).sum()
        # print(data[attribute])
        # print(data[attribute].groupby(hours).mean())
        # input()

    # keep only the unique timestamps
    matrix_data_mean = matrix_data_mean[~matrix_data_mean.index.duplicated(keep='first')]
    matrix_data_std = matrix_data_std[~matrix_data_std.index.duplicated(keep='first')]
    matrix_data_sum = matrix_data_sum[~matrix_data_sum.index.duplicated(keep='first')]

    test_matrix = matrix_data_sum - matrix_data_mean

    # plot the mean from the matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(test_matrix.T, cmap='viridis', annot=True, fmt=".2f", cbar=True, linewidths=.5)
    plt.title('Heatmap des attributs par heure de la journée (sum - mean)')
    plt.xlabel('Heures de la journée')
    plt.ylabel('Attributs')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.savefig('heatmap_test.png')
    plt.show()
    
    return

    # display the first rows of the matrix
    print(matrix_data_mean.head())
    print(matrix_data_mean.shape)

    # plot the mean from the matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(matrix_data_mean.T, cmap='viridis', annot=True, fmt=".2f", cbar=True, linewidths=.5)
    plt.title('Heatmap des attributs par heure de la journée (mean)')
    plt.xlabel('Heures de la journée')
    plt.ylabel('Attributs')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.savefig('heatmap_mean.png')    
    plt.show()

    plt.figure(figsize=(20, 10))
    sns.heatmap(matrix_data_std.T, cmap='viridis', annot=True, fmt=".2f", cbar=True, linewidths=.5)
    plt.title('Heatmap des attributs par heure de la journée (std)')
    plt.xlabel('Heures de la journée')
    plt.ylabel('Attributs')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.savefig('heatmap_std.png')
    plt.show()

    plt.figure(figsize=(20, 10))
    sns.heatmap(matrix_data_sum.T, cmap='viridis', annot=True, fmt=".2f", cbar=True, linewidths=.5)
    plt.title('Heatmap des attributs par heure de la journée (sum)')
    plt.xlabel('Heures de la journée')
    plt.ylabel('Attributs')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.savefig('heatmap_sum.png')
    plt.show()


# heatmaps()

isolation_forest_right_percent()


def old():

    # data_inc = pd.read_csv('data/real_data_ogone_incidents_truth.csv')
    # data_inc['truth'] = data_inc['truth'].astype(int)
    # data_inc['incident'] = data_inc['incident'].astype(int)

    data = pd.read_csv('data/real_data_ogone_truth.csv')
    data['truth'] = data['truth'].astype(int)

    # data = data_inc

    print(data.columns)
    headers = data.columns.values.tolist()
    headers.remove('truth')
    headers.remove('timestamp')


    # X = data[['paid_rate', 'total_transaction_count']]
    # y = data['truth']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data_truth_0 = data[data['truth'] == 0]

    # save in a variable the data with the truth label 1
    train_data_truth_1 = data[data['truth'] == 1]
    # select randomly 8 values from the train_data_truth_1 then remove them from the train_data_truth_1
    test_data_truth_1 = train_data_truth_1.sample(n=8)

    # keep 10% of the data from the train_data_truth_0
    test_data_truth_0 = train_data_truth_0.sample(frac=0.2)

    # create the training and testing data
    train_data = pd.concat([train_data_truth_1, train_data_truth_0], ignore_index=True)
    test_data = pd.concat([test_data_truth_1, test_data_truth_0], ignore_index=True)

    # shuffle the data
    train_data = train_data.sample(frac=1)
    test_data = test_data.sample(frac=1)

    X_train = train_data[headers]
    y_train = train_data['truth']
    X_test = test_data[headers]
    y_test = test_data['truth']


    print(f"Number of trues in y_train: {y_train.sum()}")
    print(f"Number of falses in y_train: {y_train.shape[0] - y_train.sum()}")
    print(f"Number of trues in y_test: {y_test.sum()}")
    print(f"Values of X_test with truth label 1: {X_test[y_test == 1]}")
    print(f"Number of falses in y_test: {y_test.shape[0] - y_test.sum()}")


    weigthed = {0: 1, 1: 6}

    print("Perceptron")
    perceptron = Perceptron(class_weight=weigthed)
    train_n_predict(X_train, y_train, X_test, y_test, perceptron)

    print("SVC")
    svm = SVC(class_weight=weigthed)
    train_n_predict(X_train, y_train, X_test, y_test, svm)

    print("Random Forest")
    rfc = RandomForestClassifier(class_weight=weigthed)
    train_n_predict(X_train, y_train, X_test, y_test, rfc)

    print("KNN")
    knn = KNeighborsClassifier(p=2, n_neighbors=2, weights='distance')
    train_n_predict(X_train, y_train, X_test, y_test, knn)


    # The following produce more than 2 classes #

    # print("Local Outlier Factor")
    # # set the local outlier factor to predict a binary label
    # lof = LocalOutlierFactor(novelty=True, contamination=0.1, n_neighbors=2, p=2)
    # lof.fit(X_train)
    # scores = lof.negative_outlier_factor_
    # y_pred = (scores < -1.5).astype(int)
    # conf_mat = confusion_matrix(y_test, y_pred)
    # print(conf_mat)

    # print("One Class SVM")
    # ocsvm = OneClassSVM()
    # ocsvm.fit(X_train)
    # scores = ocsvm.decision_function(X_test)
    # y_pred = (scores < -0.2).astype(int)
    # conf_mat = confusion_matrix(y_test, y_pred)
    # print(conf_mat)

    # print("Isolation Forest")
    # iforest = IsolationForest()
    # iforest.fit(X_train)
    # scores = iforest.decision_function(X_test)
    # y_pred = (scores < -0.2).astype(int)
    # conf_mat = confusion_matrix(y_test, y_pred)
    # print(conf_mat)

    print("Logistic Regression")
    lr = LogisticRegression(class_weight=weigthed)
    train_n_predict(X_train, y_train, X_test, y_test, lr)

    print("Decision Tree")
    dt = DecisionTreeClassifier()
    train_n_predict(X_train, y_train, X_test, y_test, dt)

    print("Gaussian Naive Bayes")
    gnb = GaussianNB()
    train_n_predict(X_train, y_train, X_test, y_test, gnb)

    print("Gradient Boosting")
    gb = GradientBoostingClassifier()
    train_n_predict(X_train, y_train, X_test, y_test, gb)

    print("Ada Boosting")
    ab = AdaBoostClassifier()
    train_n_predict(X_train, y_train, X_test, y_test, ab)

    print("MLP")
    mlp = MLPClassifier()
    train_n_predict(X_train, y_train, X_test, y_test, mlp)



    # from sklearn.inspection import DecisionBoundaryDisplay

    # disp = DecisionBoundaryDisplay.from_estimator(iforest, X_train, response_method="predict", alpha=0.5)
    # disp.ax_.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=20, edgecolor="k")
    # disp.plot(cmap="tab10", alpha=0.5)
    # disp.ax_.set_title("IsolationForest decision boundary")
    # plt.axis("tight")
    # plt.legend()
    # plt.show()


    # best = []
    # for j in range(1, 20):
    #     for i in range(10, 40):
    #         print(j, i)
    #         svm = SVC(class_weight={0: j, 1: i}, random_state=42)
    #         svm.fit(X_train, y_train)

    #         y_pred = svm.predict(X_test)

    #         # print confusion matrix
    #         tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()

    #         best.append((f1_score(y_test, y_pred), tp, fp, fn, tn, j, i))

    # best.sort(key=lambda x: x[0], reverse=True)
    # print(best[0])