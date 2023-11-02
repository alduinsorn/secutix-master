import click
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from random import randint

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, plot_tree
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression

from sklearn.metrics import confusion_matrix

from algorithms import algorithms

class Sandbox():

    def __init__(self):

        self.best_models = {} # this dictionnary will contains the models
        pass

    def compute_metrics(predict_labels, true_labels):
        # compute the confusion matrix
        cm = confusion_matrix(true_labels, predict_labels)
        # compute the precision
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        # compute the recall
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        # compute the f1 score
        f1_score = 2 * (precision * recall) / (precision + recall)
        # compute the accuracy
        accuracy = (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[0][0] + cm[1][0] + cm[0][1])

        return precision, recall, f1_score, accuracy


    def normal_training(X_train, y_train, X_test, y_test, algorithm):
        click.echo("You have selected normal training")
        click.echo("Training the model...")
        # train the model
        algorithm.train(X_train, y_train)
        # predict the labels for the test data
        predict_labels = algorithm.predict(X_test)
        # compute the metrics
        precision, recall, f1_score, accuracy = Sandbox.compute_metrics(predict_labels, y_test)
        # print the metrics
        click.echo("=========================================")
        click.echo(f"Precision: {precision}")
        click.echo(f"Recall: {recall}")
        click.echo(f"F1 score: {f1_score}")
        click.echo(f"Accuracy: {accuracy}")
        click.echo("=========================================")
    
    def fine_tuning(X_train, y_train, X_test, y_test, algorithm):
        click.echo("You have selected hyperparameters tuning")
        click.echo("Tuning the hyperparameters...")
        # train the model
        algorithm.search_best_params(X_train, y_train, X_test, y_test)
        # predict the labels for the test data
        predict_labels = algorithm.predict(X_test)
        # compute the metrics
        precision, recall, f1_score, accuracy = Sandbox.compute_metrics(predict_labels, y_test)
        # print the metrics
        click.echo("=========================================")
        click.echo(f"Precision: {precision}")
        click.echo(f"Recall: {recall}")
        click.echo(f"F1 score: {f1_score}")
        click.echo(f"Accuracy: {accuracy}")
        click.echo("=========================================")

    def main(self):
        data = pd.read_csv('./data/real_data_ogone_truth.csv')
        # data['timestamp'] = pd.to_datetime(data['timestamp'])
        # data.set_index('timestamp', inplace=True)
        data['truth'] = data['truth'].astype(int)
        # check if the data has the incident column
        if 'incident' in data.columns:
            data['incident'] = data['incident'].astype(int)

        headers = data.columns.values.tolist()
        headers.remove('truth')
        headers.remove('timestamp')

        train_data_truth_0 = data[data['truth'] == 0]
        train_data_truth_1 = data[data['truth'] == 1]
        test_data_truth_1 = train_data_truth_1.sample(n=randint(int(0.3*len(train_data_truth_1)), int(0.5*len(train_data_truth_1))))

        # keep 20% of the data from the train_data_truth_0
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



        while True:
            click.echo("Welcome to the sandbox")
            click.echo("Please select the algorithm you want to test")
            click.echo("1. Random Forest")
            click.echo("2. Gradient Boosting")
            click.echo("3. Gaussian Naive Bayes")
            click.echo("4. Decision Tree")
            click.echo("5. Logistic Regression")
            click.echo("6. Perceptron")
            click.echo("7. MLP")
            click.echo("99. Display the models")
            click.echo("0. Exit")
            choice = click.prompt("Please enter your choice", type=int)
            
            if choice == 1:
                click.echo("You have selected Random Forest")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                rf = algorithms.RandomForest()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, rf)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, rf)
                self.best_models['Random Forest'] = rf
            
            elif choice == 2:
                click.echo("You have selected Gradient Boosting")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                gb = algorithms.GradientBoosting()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, gb)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, gb)
                self.best_models['Gradient Boosting'] = gb
            
            elif choice == 3:
                click.echo("You have selected Gaussian Naive Bayes")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                gnb = algorithms.GaussianNaiveBayes()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, gnb)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, gnb)
                self.best_models['Gaussian Naive Bayes'] = gnb

            elif choice == 4:
                click.echo("You have selected Decision Tree")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                dt = algorithms.DecisionTree()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, dt)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, dt)
                self.best_models['Decision Tree'] = dt

            elif choice == 5:
                click.echo("You have selected Logistic Regression")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                lr = algorithms.LogisticRegression()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, lr)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, lr)
                self.best_models['Logistic Regression'] = lr

            elif choice == 6:
                click.echo("You have selected Perceptron")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                p = algorithms.Perceptron()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, p)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, p)
                self.best_models['Perceptron'] = p

            elif choice == 7:
                click.echo("You have selected MLP")
                click.echo("Please select the training mode")
                click.echo("1. Normal training")
                click.echo("2. Hyperparameters tuning")
                choice = click.prompt("Please enter your choice", type=int)
                mlp = algorithms.MLP()
                if choice == 1:
                    Sandbox.normal_training(X_train, y_train, X_test, y_test, mlp)
                elif choice == 2:
                    Sandbox.fine_tuning(X_train, y_train, X_test, y_test, mlp)
                self.best_models['MLP'] = mlp

            elif choice == 99:
                click.echo("You have selected to display the models")
                for model in self.best_models:
                    click.echo(f"{model}: {self.best_models[model]}")

            elif choice == 0:
                click.echo("You have selected to exit the sandbox")
                break


            else:
                click.echo("Invalid choice, please try again")
                continue

            
    # data = pd.read_csv('./data/real_data_ogone_truth.csv')
    # data['timestamp'] = pd.to_datetime(data['timestamp'])
    # data.set_index('timestamp', inplace=True)

    # jan_month = data.loc['2023-01-01':'2023-01-31']
    # x_ticks_labels = [f'h{i}' for i in range(24)]
    # # get the min and the max of the paid_rate
    # min_paid_rate = jan_month['paid_rate'].min()-1
    # max_paid_rate = jan_month['paid_rate'].max()+1
    # # for every day in the month of january plot the paid_rate
    # for i in range(1, 32):
    #     plt.figure(figsize=(20, 10))
    #     day = jan_month.loc[f'2023-01-{i}']
    #     x_ticks = [datetime(2023, 1, i, j, 0) for j in range(24)]
    #     plt.plot(day['paid_rate'])
    #     plt.xticks(x_ticks, x_ticks_labels)
    #     plt.title(f"January {i}th")
    #     plt.ylim(min_paid_rate, max_paid_rate)

    #     hour6 = datetime(2023, 1, i, 6, 0)
    #     plt.axvline(hour6, color='r', linestyle='--')
    #     plt.grid()
    #     plt.show()


if __name__ == "__main__":
    sandbox = Sandbox()
    sandbox.main()