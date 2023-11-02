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