import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime
import os
import random
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but IsolationForest was fitted with feature names", category=UserWarning)


def percent_analysis(unpaid, paid):
    print(f"{(unpaid / (unpaid + paid) * 100):.2f}% ({unpaid+paid})")

def prepare_data_raw(data, onehot_state=False):
    # convert the date into columns for year, month, day, hour, minute
    data['year'] = data['CREATION_DATE'].dt.year
    data['month'] = data['CREATION_DATE'].dt.month
    data['day'] = data['CREATION_DATE'].dt.day
    data['hour'] = data['CREATION_DATE'].dt.hour
    data['minute'] = data['CREATION_DATE'].dt.minute
    data['second'] = data['CREATION_DATE'].dt.second

    data = data[data['PAYMENT_TYPE'] == 'OGONE_HID']

    # print(data['PAYMENT_METHOD'].value_counts())
    # input()

    # drop some columns  
    data = data.drop(['CREATION_DATE', 'PAYMENT_ID', 'PREVIOUS_STATES', 'PAYMENT_TYPE', 'ISSUER'], axis=1)
    # data = data.drop([ 'PAYMENT_ID', 'PREVIOUS_STATES', 'PAYMENT_TYPE', 'ISSUER'], axis=1)


    if onehot_state:
        # PAID, UNPAID, REFUNDED, REFUSED, ABANDONED, CANC_ADMIN
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('CANC_ADMIN', 'ABANDONED')
        # keep only the rows where the payment state is in ['PAID', 'UNPAID', 'REFUNDED', 'REFUSED', 'ABANDONED']
        data = data[data['PAYMENT_STATE'].isin(['PAID', 'UNPAID', 'REFUNDED', 'REFUSED', 'ABANDONED'])]
        data = pd.get_dummies(data, columns=['PAYMENT_STATE'])
    else:
        # # convert 'STATE' column to numeric 'PAID' = 1, 'UNPAID' = 0, 'ABANDONED' = 0
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('PAID', 1)
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('UNPAID', 0)
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('ABANDONED', 0)
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('REFUSED', 0)
        data['PAYMENT_STATE'] = data['PAYMENT_STATE'].replace('REFUNDED', 0)
        # keep only the rows where the payment state is 0 or 1
        data = data[data['PAYMENT_STATE'].isin([0, 1])]

    # for each payment_method count the number of transactions
    payment_method = data['PAYMENT_METHOD'].value_counts()
    # drop the payment method where the number of transactions is less than 1000
    data = data[~data['PAYMENT_METHOD'].isin(payment_method[payment_method < 1000].index)]
    
    # convert the categorical column PAYMENT_METHOD to numerical or one-hot encoding
    data = pd.get_dummies(data, columns=['PAYMENT_METHOD'])
    
    return data

def load_data_high_level(fname):
    data = pd.read_csv(fname)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    return data

def prepare_data_high_level(data, percent_only=False, with_date=False):

    if with_date:
        # # convert the date into columns for year, month, day, hour, minute, second
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['second'] = data.index.second

        # normalize year, month, day, hour, minute, second
        scaler_year = StandardScaler()
        data['year'] = scaler_year.fit_transform(data['year'].values.reshape(-1, 1))
        scaler_month = StandardScaler()
        data['month'] = scaler_month.fit_transform(data['month'].values.reshape(-1, 1))
        scaler_day = StandardScaler()
        data['day'] = scaler_day.fit_transform(data['day'].values.reshape(-1, 1))
        scaler_hour = StandardScaler()
        data['hour'] = scaler_hour.fit_transform(data['hour'].values.reshape(-1, 1))
        scaler_minute = StandardScaler()
        data['minute'] = scaler_minute.fit_transform(data['minute'].values.reshape(-1, 1))
        scaler_second = StandardScaler()
        data['second'] = scaler_second.fit_transform(data['second'].values.reshape(-1, 1))

    if percent_only:
        data = data.drop(['total_transaction_count'], axis=1)

    norm_data = data.copy()
    scaler_percentage = StandardScaler()
    norm_data['percentage'] = scaler_percentage.fit_transform(norm_data['percentage'].values.reshape(-1, 1))

    scaler_transaction_count = StandardScaler()
    if not percent_only:
        norm_data['total_transaction_count'] = scaler_transaction_count.fit_transform(norm_data['total_transaction_count'].values.reshape(-1, 1))

    test_data = norm_data['2023-07-01 00:00:00':'2023-11-28 23:59:59']
    norm_data = norm_data.drop(test_data.index)

    return norm_data, test_data, (scaler_percentage, scaler_transaction_count)

def load_anomalies_nicely(fname):
    ANOMALIES_START_DATE = '2023-07-01 00:00:00'
    ANOMALIES_END_DATE = '2023-11-28 23:59:59'
    # Load anomalies dataframe
    anomalies = pd.read_csv(fname)
    anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
    anomalies = anomalies.set_index('timestamp')
    anomalies.index = pd.DatetimeIndex(anomalies.index.values, freq=anomalies.index.inferred_freq)
    anomalies = anomalies[ANOMALIES_START_DATE:ANOMALIES_END_DATE]
    anomalies = anomalies[anomalies['status'] == True]
    anomalies['status'] = -1
    # Create non-anomalies dataframe
    # non_anomalies = pd.DataFrame(pd.date_range(ANOMALIES_START_DATE, ANOMALIES_END_DATE, freq='H'), columns=['timestamp'])
    # non_anomalies = non_anomalies.set_index('timestamp')
    # non_anomalies.index = pd.DatetimeIndex(non_anomalies.index.values, freq=non_anomalies.index.inferred_freq)
    # non_anomalies = non_anomalies.drop(anomalies.index)
    # non_anomalies['status'] = False
    # # Concatenate anomalies and non-anomalies
    # anomalies = pd.concat([anomalies, non_anomalies])
    anomalies.sort_index(inplace=True)
    anomalies.rename(columns={'status': 'predictions'}, inplace=True)
    print(anomalies['predictions'].value_counts())
    return anomalies


def kmeans_transactions(fname, cluster_number, start_datetime, end_datetime, onehot_state=False, savefig=False):
    data = pd.read_csv(fname)
    data['CREATION_DATE'] = pd.to_datetime(data['CREATION_DATE'])
    # print(data.columns)

    # print(data['PAYMENT_TYPE'].value_counts())
    # print(data['PAYMENT_STATE'].value_counts())
    # print(data['PAYMENT_METHOD'].value_counts())
    # exit()


    # # display the number of transaction for every minutes
    # data['MINUTES'] = data['CREATION_DATE'].dt.minute
    # data_incident = data[(data['CREATION_DATE'] > datetime.datetime(2023, 1, 17, 10, 0)) & (data['CREATION_DATE'] < datetime.datetime(2023, 1, 17, 11, 0))]
    # print(data_incident['MINUTES'].value_counts().sort_index())

    # # make a bar plot of the number of transactions for every minutes
    # plt.bar(data_incident['MINUTES'].value_counts().sort_index().index, data_incident['MINUTES'].value_counts().sort_index().values)
    # plt.xlabel('Minutes')
    # plt.ylabel('Number of transactions')
    # plt.title('Number of transactions for every minutes during the incident')
    # # plt.savefig('number_transactions_per_minutes.png')
    # plt.show()
    # input()

    # data_incident = data[(data['CREATION_DATE'] > datetime.datetime(2023, 1, 17, 10, 0)) & (data['CREATION_DATE'] < datetime.datetime(2023, 1, 17, 11, 0))]
    # print(len(data_incident))
    # print(data_incident['PAYMENT_STATE'].value_counts())
    # print(data_incident['PAYMENT_METHOD'].value_counts())
    # print(data_incident['ISSUER'].value_counts())
    # input()


    norm_data = prepare_data_raw(data, onehot_state)
    # print(norm_data.head())
    # print(norm_data.columns)

    # ### Elbow method ###
    # sum_squared_distances = []
    # for num_clusters in range(2, 10):
    #     kmeans = KMeans(n_clusters=num_clusters)
    #     kmeans.fit(norm_data)
    #     # print(num_clusters, kmeans.inertia_)
    #     sum_squared_distances.append(kmeans.inertia_)

    # plt.plot(range(2, 10), sum_squared_distances)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Sum of Squared Distances')
    # plt.title('Elbow Method based on the transactions data transformed')
    # plt.savefig('elbow_method.png')
    # plt.show()
    # return

    # recreate the datetime using the year, month, day, hour, minute, second to keep only the data between start_datetime and end_datetime
    norm_data['CREATION_DATE'] = pd.to_datetime(norm_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

    display_data = norm_data[(norm_data['CREATION_DATE'] > start_datetime) & (norm_data['CREATION_DATE'] < end_datetime)]
    display_data = display_data.drop(['CREATION_DATE'], axis=1)
    print("display_data", display_data.head())

    # # create a kmeans model
    kmeans = KMeans(n_clusters=cluster_number, random_state=42)
    # remove the display_data from the norm_data
    norm_data = norm_data.drop(display_data.index)
    norm_data = norm_data.drop(['CREATION_DATE'], axis=1)

    print("norm_data", norm_data.head())
    kmeans.fit(norm_data)

    # # predict the clusters for the data
    display_data['LABEL'] = kmeans.predict(display_data)

    # recreate the datetime using the year, month, day, hour, minute, second for the plot
    display_data['CREATION_DATE'] = pd.to_datetime(display_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

    if onehot_state:
        # recreate the payment_state
        display_data['PAYMENT_STATE'] = ''
        for index, row in display_data.iterrows():
            for column in display_data.columns:
                if row[column] == 1 and column.startswith('PAYMENT_STATE'):
                    # remove PAYMENT_STATE from the name
                    display_data.at[index, 'PAYMENT_STATE'] = column.replace('PAYMENT_STATE_', '')
                    break
        
        # # create a column with the payment state as a number
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE'].replace('PAID', 0)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('UNPAID', 1)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('REFUNDED', 2)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('REFUSED', 3)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('ABANDONED', 4)


    # for each cluster compute the number of paid and unpaid transactions
    for cluster in range(cluster_number):
        cluster_data = display_data[display_data['LABEL'] == cluster]
        print(cluster_data['PAYMENT_STATE'].value_counts())

    # recreate the payment_method fields for the plot
    display_data['PAYMENT_METHOD'] = ''
    for index, row in display_data.iterrows():
        for column in display_data.columns:
            if row[column] == 1 and column.startswith('PAYMENT_METHOD'):
                # remove PAYMENT_METHOD from the name
                display_data.at[index, 'PAYMENT_METHOD'] = column.replace('PAYMENT_METHOD_', '')
                break
    
    folder_name = f'kmeans/{cluster_number}_clusters/{"onehot_payment_state" if onehot_state else "binary_payment_state"}'

    if onehot_state:
        for i in range(cluster_number):
            display_data_spec = display_data[display_data['LABEL'] == i]

            plt.figure(figsize=(20, 10))
            for payment_state in ['PAID', 'UNPAID', 'REFUNDED', 'REFUSED', 'ABANDONED']:
                display_data_spec_payment_state = display_data_spec[display_data_spec['PAYMENT_STATE'] == payment_state]
                count_transactions = len(display_data_spec_payment_state)
                plt.scatter(display_data_spec_payment_state['CREATION_DATE'], display_data_spec_payment_state['PAYMENT_METHOD'], label=f'{payment_state} ({count_transactions})')
            
            plt.xlabel('Payment state')
            plt.ylabel('Payment methods')
            plt.legend()


            if start_datetime.hour == 10 and start_datetime.day == 17:
                plt.title(f'Data points by payment methods for cluster {i} during the {start_datetime.strftime("%dth/%b")} incident period')
            else:
                plt.title(f'Data points by payment methods for cluster {i} during the {start_datetime.strftime("%dth/%b")} normal period')
            if savefig: plt.savefig(f'{folder_name}/transactions/kmeans_{start_datetime.strftime("%dth_%Hh")}_january_2023_cluster_{i}_with_payment_state.png')
            plt.show()
    else:
        for i in range(cluster_number):
            display_data_spec = display_data[display_data['LABEL'] == i]

            display_data_spec_paid = display_data_spec[display_data_spec['PAYMENT_STATE'] == 1]
            display_data_spec_unpaid = display_data_spec[display_data_spec['PAYMENT_STATE'] == 0]

            plt.figure(figsize=(20, 10))
            plt.scatter(display_data_spec_paid['CREATION_DATE'], display_data_spec_paid['PAYMENT_METHOD'], c='blue')
            plt.scatter(display_data_spec_unpaid['CREATION_DATE'], display_data_spec_unpaid['PAYMENT_METHOD'], c='red')
            plt.xlabel('Creation date')
            plt.ylabel('Payment methods')
            # add special legend containing the number of paid and unpaid transactions
            paid_transaction_count = len(display_data_spec[display_data_spec['PAYMENT_STATE'] == 1])
            unpaid_transaction_count = len(display_data_spec[display_data_spec['PAYMENT_STATE'] == 0])
            plt.legend([f'Paid: {paid_transaction_count}', f'Unpaid: {unpaid_transaction_count}'])

            percent_analysis(unpaid_transaction_count, paid_transaction_count)
            
            if start_datetime.hour == 10 and start_datetime.day == 17:
                plt.title(f'Data points by payment methods for cluster {i} during the {start_datetime.strftime("%dth/%b")} incident period')
            else:
                plt.title(f'Data points by payment methods for cluster {i} during the {start_datetime.strftime("%dth/%b")} normal period')
            if savefig: plt.savefig(f'{folder_name}/transactions/kmeans_{start_datetime.strftime("%dth_%Hh")}_january_2023_cluster_{i}.png')
            plt.show()
    


    # # # plot some of the data points
    # plt.scatter(display_data['CREATION_DATE'], display_data['PAYMENT_STATE'], c=display_data_labels, cmap='rainbow')
    # plt.xlabel('Payment State')
    # plt.ylabel('Amount')
    # plt.show()

####### K-means with high level data ######

def elbow_method(data):
    sum_squared_distances = []
    for num_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        # print(num_clusters, kmeans.inertia_)
        sum_squared_distances.append(kmeans.inertia_)

    print(sum_squared_distances)
    # print the difference between the actual and the precedent value
    for i in range(1, len(sum_squared_distances)):
        print(sum_squared_distances[i] - sum_squared_distances[i-1])

    plt.plot(range(2, 10), sum_squared_distances)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method based on the transactions data transformed')
    plt.savefig('elbow_method.png')
    plt.grid()
    plt.show()

def plot_data_high_level_kmeans(data, savefig=False, percent_only=False):
    cluster_number = data['LABEL'].max()+1
    # plot the data points for the 18th september 2023 to the 20th september 2023, but with the datetime on the x axis
    plt.figure(figsize=(20, 10))
    for i in range(cluster_number):
        display_data_spec = data[data['LABEL'] == i]
        plt.scatter(display_data_spec.index, display_data_spec['percentage'], label=f'{len(display_data_spec)}')
    plt.xlabel('Datetime')
    plt.ylabel('Percentage')
    plt.legend()
    plt.title(f'Data points by percentage and date for {cluster_number} clusters')
    if savefig: 
        # get the min date
        min_date = data.index.min().strftime("%dth")
        max_date = data.index.max().strftime("%dth")
        date_text = f"{min_date}_to_{max_date}"
        percent_only_text = "_percent_only" if percent_only else ""
        plt.savefig(f'kmeans/high_level/{cluster_number}_clusters/{date_text}_kmeans_high_level{percent_only_text}.png')
    plt.show()

def kmeans_high_level(train_data, test_data, scalers, cluster_number, percent_only=True, savefig=False, with_date=False, random_state=42, plot_data=False):
    ### Old part, before the kfold cross validation ###
    # data = load_data_high_level(fname)
    # train_data, test_data, scalers = prepare_data_high_level(data, percent_only=percent_only, with_date=with_date)

    # elbow_method(train_data)
    # exit()

    kmeans = KMeans(n_clusters=cluster_number, random_state=random_state)
    kmeans.fit(train_data)

    # predict the clusters for the data
    test_data['predictions'] = kmeans.predict(test_data)

    # convert the percentage and the total_transaction_count to their original value
    test_data['percentage'] = scalers[0].inverse_transform(test_data['percentage'].values.reshape(-1, 1))
    if not percent_only:
        test_data['total_transaction_count'] = scalers[1].inverse_transform(test_data['total_transaction_count'].values.reshape(-1, 1))

    if plot_data:
        ## Anomalies
        # 2023-09-06 09:00:00
        # 2023-09-17 15:00:00
        # 2023-09-17 16:00:00
        # 2023-09-19 09:00:00
        display_data1 = test_data['2023-09-05 00:00:00':'2023-09-07 23:59:59'].copy()
        display_data2 = test_data['2023-09-16 00:00:00':'2023-09-18 23:59:59'].copy()
        display_data3 = test_data['2023-09-18 00:00:00':'2023-09-20 23:59:59'].copy()

        plot_data_high_level_kmeans(display_data1, savefig=savefig, percent_only=percent_only)
        plot_data_high_level_kmeans(display_data2, savefig=savefig, percent_only=percent_only)
        plot_data_high_level_kmeans(display_data3, savefig=savefig, percent_only=percent_only)

        # plt.figure(figsize=(20, 10))
        # plt.scatter(test_data['percentage'], test_data['total_transaction_count'], c=test_data['predictions'], cmap='rainbow')
        # plt.xlabel('Percentage')
        # plt.ylabel('Total transaction count')
        # plt.title(f'Representation of the data points by percentage and total transaction count with different colors for each cluster')
        # plt.savefig(f'kmeans/high_level/{cluster_number}_clusters/representation_kmeans_high_level.png')
        # plt.show()

    # print(test_data['predictions'].value_counts())

    # save into a list the number of data points for each cluster, use the list index as the cluster number
    cluster_number_list = [len(test_data[test_data['predictions'] == i]) for i in range(cluster_number)]
    lowest_cluster_index = cluster_number_list.index(min(cluster_number_list))
    # change the lowest cluster index to -1
    test_data['predictions'] = test_data['predictions'].replace(lowest_cluster_index, -1)

    # print the values for the lowest cluster
    # anomalies_data = test_data[test_data['predictions'] == lowest_cluster_index].head(min(cluster_number_list))
    # print(anomalies_data)

    return test_data

####### End K-means with high level #######

def isolation_level_transactions(fname, n_estimator, start_datetime, end_datetime, onehot_state=False, savefig=False):
    data = pd.read_csv(fname)
    data['CREATION_DATE'] = pd.to_datetime(data['CREATION_DATE'])
    
    # display the number of values during the incident period
    # print(len(data[(data['CREATION_DATE'] > start_datetime) & (data['CREATION_DATE'] < end_datetime)]))
    # print(len(data))
    # exit()

    norm_data = prepare_data_raw(data, onehot_state)

    # print(norm_data.columns)
    # exit()

    norm_data['CREATION_DATE'] = pd.to_datetime(norm_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

    display_data = norm_data[(norm_data['CREATION_DATE'] > start_datetime) & (norm_data['CREATION_DATE'] < end_datetime)]
    display_data = display_data.drop(['CREATION_DATE'], axis=1)
    print("display_data", display_data.head())

    ### Contamination value resolved by : 70% of the data of the incident period are anomaly so -> (0.4*1924) / 458181 = 0.00167 -> give no anomaly
    isolation_forest = IsolationForest(n_estimators=n_estimator, max_samples='auto', contamination=0.167, random_state=42)

    # remove the display_data from the norm_data
    norm_data = norm_data.drop(display_data.index)
    norm_data = norm_data.drop(['CREATION_DATE'], axis=1)

    isolation_forest.fit(norm_data)

    display_data['predictions'] = isolation_forest.predict(display_data)
    # display_data['predictions'] = isolation_forest.predict(display_data.drop(['CREATION_DATE'], axis=1))
    
    # recreate the datetime using the year, month, day, hour, minute, second for the plot
    display_data['CREATION_DATE'] = pd.to_datetime(display_data[['year', 'month', 'day', 'hour', 'minute', 'second']])
    
    if onehot_state:
        # recreate the payment_state
        display_data['PAYMENT_STATE'] = ''
        for index, row in display_data.iterrows():
            for column in display_data.columns:
                if row[column] == 1 and column.startswith('PAYMENT_STATE'):
                    # remove PAYMENT_STATE from the name
                    display_data.at[index, 'PAYMENT_STATE'] = column.replace('PAYMENT_STATE_', '')
                    break
        
        # # create a column with the payment state as a number
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE'].replace('PAID', 0)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('UNPAID', 1)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('REFUNDED', 2)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('REFUSED', 3)
        display_data['PAYMENT_STATE_NUM'] = display_data['PAYMENT_STATE_NUM'].replace('ABANDONED', 4)

    
    display_data['PAYMENT_METHOD'] = ''
    for index, row in display_data.iterrows():
        for column in display_data.columns:
            if row[column] == 1 and column.startswith('PAYMENT_METHOD'):
                # remove PAYMENT_METHOD from the name
                display_data.at[index, 'PAYMENT_METHOD'] = column.replace('PAYMENT_METHOD_', '')
                break

    folder_name = f'isolation_forest/{n_estimator}_trees/{"onehot_payment_state" if onehot_state else "binary_payment_state"}'

    print(display_data['predictions'].value_counts())
    # print(display_data.head())

    # display the number of anomaly and normal point for each payment_method
    for payment_method in display_data['PAYMENT_METHOD'].unique():
        display_data_payment_method = display_data[display_data['PAYMENT_METHOD'] == payment_method]
        print(f"### payment_method {payment_method} ###")
        print(display_data_payment_method['predictions'].value_counts())
        print()

    if onehot_state:
        plt.figure(figsize=(20, 10))
        for i in [-1, 1]:
            display_data_spec = display_data[display_data['predictions'] == i]
            count_transactions = len(display_data_spec)
            plt.scatter(display_data_spec['CREATION_DATE'], display_data_spec['PAYMENT_METHOD'], label=f'{"Anomaly" if i == -1 else "Not anomaly"} ({count_transactions})')
        plt.xlabel('Creation date')
        plt.ylabel('Payment methods')
        plt.legend()
        plt.title(f'Data points by payment methods during the {start_datetime.strftime("%dth/%b")} incident period')
        if savefig: plt.savefig(f'{folder_name}/isolation_forest_{start_datetime.strftime("%dth_%Hh")}_january_2023.png')
        plt.show()
    else:
        # create a new column for each possibility (case 1 (paid and anomaly), case 2 (paid and not anomaly), case 3 (unpaid and anomaly), case 4 (unpaid and not anomaly)
        display_data['CASE'] = ''
        for index, row in display_data.iterrows():
            if row['PAYMENT_STATE'] == 0 and row['predictions'] == -1:
                display_data.at[index, 'CASE'] = "PAID + ANOMALY"
            elif row['PAYMENT_STATE'] == 0 and row['predictions'] == 1:
                display_data.at[index, 'CASE'] = "PAID + NOT ANOMALY"
            elif row['PAYMENT_STATE'] == 1 and row['predictions'] == -1:
                display_data.at[index, 'CASE'] = "UNPAID + ANOMALY"
            elif row['PAYMENT_STATE'] == 1 and row['predictions'] == 1:
                display_data.at[index, 'CASE'] = "UNPAID + NOT ANOMALY"

        colors = ['orange', 'blue', 'red', 'green']

        # plot the data points based on the creation_date and the payment_method
        plt.figure(figsize=(20, 10))
        for i, case in enumerate(['PAID + ANOMALY', 'PAID + NOT ANOMALY', 'UNPAID + ANOMALY', 'UNPAID + NOT ANOMALY']):
            display_data_case = display_data[display_data['CASE'] == case]
            count_transactions = len(display_data_case)
            print(f"case {case} ({count_transactions})")
            plt.scatter(display_data_case['CREATION_DATE'], display_data_case['PAYMENT_METHOD'], label=f'{case} ({count_transactions})', c=colors[i])
        plt.xlabel('Creation date')
        plt.ylabel('Payment methods')
        plt.legend()
        plt.title(f'Data points by payment methods during the {start_datetime.strftime("%dth/%b")} incident period')
        if savefig: plt.savefig(f'{folder_name}/isolation_forest_{start_datetime.strftime("%dth_%Hh")}_january_2023.png')
        plt.show()

def isolation_high_level(data_train, data_test, scalers, n_estimator, contamination=0.01, percent_only=True, savefig=False, with_date=False, random_state=42, plot_data=False):
    ### Old part, before the kfold cross validation ###
    # data = load_data_high_level(fname)
    # data_train, data_test, scalers = prepare_data_high_level(data, percent_only=percent_only, with_date=with_date)

    # contamination d'environ 0.1% pour les anomalies
    isolation_forest = IsolationForest(n_estimators=n_estimator, max_samples='auto', contamination=contamination, random_state=random_state)
    isolation_forest.fit(data_train)
    data_test['predictions'] = isolation_forest.predict(data_test)
    # print(data_test['predictions'].value_counts())

    # convert the percentage and the total_transaction_count to their original value
    data_test['percentage'] = scalers[0].inverse_transform(data_test['percentage'].values.reshape(-1, 1))
    if not percent_only:
        data_test['total_transaction_count'] = scalers[1].inverse_transform(data_test['total_transaction_count'].values.reshape(-1, 1))

    if plot_data:
        plt.figure(figsize=(20, 10))
        plt.scatter(data_test.index, data_test['percentage'], c=data_test['predictions'], cmap='rainbow')
        plt.xlabel('Datetime')
        plt.ylabel('Percentage')
        plt.title(f'Data points by percentage and date for {n_estimator} trees')
        if savefig:
            min_date = data_test.index.min().strftime("%dth")
            max_date = data_test.index.max().strftime("%dth")
            date_text = f"{min_date}_to_{max_date}"
            percent_only_text = "_percent_only" if percent_only else ""
            with_date_text = "_with_date" if with_date else ""
            plt.savefig(f'isolation_forest/high_level/{n_estimator}_trees/{date_text}_isolation_forest{percent_only_text}{with_date_text}.png')
        plt.show()

    # anomalies_data = data_test[data_test['predictions'] == -1]
    # print(anomalies_data)
    # anomalies = load_anomalies_nicely('./data/anomalies_ogone.csv')
    # val = compute_score(anomalies, data_test)
    # if val <= 7:
    #     # save the model
    #     with open(f'isolation_forest/high_level/{n_estimator}_trees/model.pkl', 'wb') as f:
    #         pickle.dump(isolation_forest, f)

    return data_test


def compute_score(anomalies, predictions):
    # test if predictions is empty
    if len(predictions) == 0:
        return len(anomalies)

    # check the value 'predictions' of the predictions dataframe to verify that the point is an anomaly or not (check if -1 for anomalies))
    tp = []; fp = []; fn = []; tn = []
    score = 0
    # compute the true positive and false positive
    for index, row in anomalies.iterrows():
        try :
            if predictions.loc[index, 'predictions'] == -1:
                tp.append(index)
            else:
                fp.append(index)
        except:
            fp.append(index)
    
    # compute the score
    for fpfp in fp: # check if the false positive is between 1am and 6am -> count 2.5
        if fpfp.hour >= 1 and fpfp.hour <= 6:
            score += 2.5
        else:
            score += 1
    
    # compute the false negative and true negative
    for index, row in predictions.iterrows():
        if row['predictions'] == -1 and index not in tp:
            fn.append(index)
    score += len(fn)

    return score

### Part for the transaction analysis ###
# their is an incident the 17th of january 2023 between 10 and 11 am
incident_start_datetime = datetime.datetime(2023, 1, 17, 10, 0)
incident_end_datetime = datetime.datetime(2023, 1, 17, 11, 0)
normal_start_datetime = datetime.datetime(2023, 1, 18, 10, 0)
normal_end_datetime = datetime.datetime(2023, 1, 18, 11, 0)
night_start_datetime = datetime.datetime(2023, 1, 18, 3, 0)
night_end_datetime = datetime.datetime(2023, 1, 18, 4, 0)

other_normal_start_datetime = datetime.datetime(2023, 1, 19, 13, 0)
other_normal_end_datetime = datetime.datetime(2023, 1, 19, 14, 0)


def analysis_kmeans():
    savefig = True

    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, incident_start_datetime, incident_end_datetime, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, normal_start_datetime, normal_end_datetime, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, night_start_datetime, night_end_datetime, savefig=savefig)

    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, incident_start_datetime, incident_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, normal_start_datetime, normal_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 3, night_start_datetime, night_end_datetime, onehot_state=True, savefig=savefig)

    kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, incident_start_datetime, incident_end_datetime, savefig=savefig)
    kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, normal_start_datetime, normal_end_datetime, savefig=savefig)
    kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, night_start_datetime, night_end_datetime, savefig=savefig)
    kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, other_normal_start_datetime, other_normal_end_datetime, savefig=savefig)

    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, incident_start_datetime, incident_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, normal_start_datetime, normal_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 2, night_start_datetime, night_end_datetime, onehot_state=True, savefig=savefig)


    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, incident_start_datetime, incident_end_datetime, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, normal_start_datetime, normal_end_datetime, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, night_start_datetime, night_end_datetime, savefig=savefig)

    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, incident_start_datetime, incident_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, normal_start_datetime, normal_end_datetime, onehot_state=True, savefig=savefig)
    # kmeans_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', 4, night_start_datetime, night_end_datetime, onehot_state=True, savefig=savefig)


def analysis_isolation_forest():
    savefig = True
    for estimator in [10, 50]:
        isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', estimator, incident_start_datetime, incident_end_datetime, onehot_state=True, savefig=savefig)
        isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', estimator, normal_start_datetime, normal_end_datetime, onehot_state=True, savefig=savefig)

        isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', estimator, incident_start_datetime, incident_end_datetime, onehot_state=False, savefig=savefig)
        isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', estimator, normal_start_datetime, normal_end_datetime, onehot_state=False, savefig=savefig)


def plot_score_for_finding_problems(data):
    # from the month of july to november, 3 days by 3 days
    data = data['percentage']
    days_in_month = [31, 31, 30, 31, 28]
    months_names = ['july', 'august', 'september', 'october', 'november']
    for j, days in enumerate(days_in_month):
        if not os.path.exists(f'search_anomalies/{months_names[j]}'):
            os.mkdir(f'search_anomalies/{months_names[j]}')
        # plot 3 days by 3 days
        for i in range(1, days, 3):
            print(i, months_names[j])
            plt.figure(figsize=(20, 10))
            # check if the i + 2th day is 30 -> if yes plot also the 31th day
            if i + 3 >= days and days == 31:
                plt.plot(data[f'2023-{j+7:02}-{i:02}':f'2023-{j+7:02}-{i+3:02}'])
            else:
                plt.plot(data[f'2023-{j+7:02}-{i:02}':f'2023-{j+7:02}-{i+2:02}'])
            plt.xlabel('Datetime')
            plt.ylabel('Percentage')
            if i + 3 >= days and days == 31:
                plt.title(f'Percentage for the {i}th, {i+1}th, {i+2}th and {i+3}th of {months_names[j]} 2023')
            else:
                plt.title(f'Percentage for the {i}th, {i+1}th and {i+2}th of {months_names[j]} 2023')
            
            # plot a vertical line for each beginning of the day
            for k in range(4 if (i + 3 >= days and days==31) else 3):
                plt.axvline(x=datetime.datetime(2023, j+7, i+k, 0, 0), color='red', linestyle='--')
                plt.axvline(x=datetime.datetime(2023, j+7, i+k, 7, 0), color='green', linestyle='--')

            # plt.savefig(f'search_anomalies/{months_names[j]}/{i}th_{i+1}th_{i+2}th_{months_names[j]}_2023.png')
            plt.show()
            plt.close()


def kfold_cross_validation(data, anomalies, fold_number, algorithm_name, algorithm_function, function_parameter, random_state=None):
    fold_score = []
    for i in range(fold_number):
        data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)

        data_train_portion = len(data_train) // fold_number
        # the dataset is expanding, after each fold -> idea of a time series because the test data is the 5 last month of the dataset
        if i == fold_number - 1:
            fold_data_train = data_train
        else:
            fold_data_train = data_train[:(i+1)*data_train_portion]

        # results = kmeans_high_level(fold_data_train, data_test, scalers, 3, percent_only=True, savefig=False, with_date=False, random_state=None, plot_data=False)
        results = algorithm_function(fold_data_train, data_test, scalers, function_parameter, random_state=random_state)
        # save the score and the number of data points used
        fold_score.append((compute_score(anomalies, results), len(fold_data_train)))

    text_margin = 0.15
    plt.figure(figsize=(20, 10))
    for i in range(fold_number):
        plt.bar(i+1, fold_score[i][0], label=f'Fold {i+1} ({fold_score[i][0]})')
        plt.text(i+1-text_margin, fold_score[i][0]+text_margin, f'{fold_score[i][1]}')
    plt.xlabel('Fold number')
    plt.ylabel('Score (Extra hours)')
    plt.title(f'Score for each fold for the {algorithm_name} algorithm with the number of data points used on top of each bar')
    plt.legend()
    # plt.savefig(f'kfold_cross_validation/{algorithm_name}.png')
    plt.show()

def find_best_model_isolation_forest(data, anomalies):
    data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)

    isolation_high_level(data_train, data_test, scalers)
    pass

def algorithms_test_one_versus_one(data, anomalies):
    # train 5 models of each algorithm and display the score for each model
    kmeans_cluster = 3
    iso_trees = 50
    iso_contamination = 0.01

    kmeans_score = []
    isolation_score = []
    for i in range(5):
        data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)
        safe_data_train = data_train.copy()
        safe_data_test = data_test.copy()
        res_kmeans = kmeans_high_level(data_train, data_test, scalers, kmeans_cluster, percent_only=True, savefig=False, with_date=False, random_state=None, plot_data=False)
        res_isolation = isolation_high_level(safe_data_train, safe_data_test, scalers, iso_trees, contamination=iso_contamination, random_state=None, plot_data=False)

        kmeans_score.append(compute_score(anomalies, res_kmeans))
        isolation_score.append(compute_score(anomalies, res_isolation))

    print("kmeans_score", kmeans_score)
    print("isolation_score", isolation_score)

    # plot in a bar chart the score for each algorithm, set the x axis as the name of each algorithm + the number of the model
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.bar(i+1, kmeans_score[i], label=f'K-means {i+1} ({kmeans_score[i]})', width=0.2, color='orange')
        plt.bar(i+1+0.2, isolation_score[i], label=f'Isolation Forest {i+1} ({isolation_score[i]})', width=0.2, color='blue')


    plt.xlabel('Model number')
    plt.ylabel('Score (Extra hours)')
    plt.title(f'Score for each model for the K-means and Isolation Forest algorithm')
    plt.legend()
    # plt.savefig(f'kfold_cross_validation/kmeans_vs_isolation.png')
    plt.show()


def threshold_model(data_test, anomalies):
    results = []
    for tresh in range(101):
        data_test_copy = data_test.copy()
        # create a new column 'predictions' that is -1 if the 'percentage' is below the threshold and 1 if above
        data_test_copy['predictions'] = data_test_copy['percentage'].apply(lambda x: -1 if x < tresh else 1)
        # print(data_test_copy.head())
        # input()
        results.append(compute_score(anomalies, data_test_copy))

    plt.figure(figsize=(20, 10))
    plt.plot(range(101), results)
    plt.xlabel('Threshold')
    plt.ylabel('Score (Extra hours)')
    plt.title(f'Score for each threshold for the threshold model')
    # plt.savefig(f'kfold_cross_validation/threshold_model.png')
    plt.show()

    print(f"Minimum score {min(results)} for the threshold {results.index(min(results))}")
    



### Analysis of the algorithms behavior ###
# kmeans_high_level('./data/2years_ogone_noise_reduction.csv', 3, percent_only=True, savefig=False)
# kmeans_high_level('./data/2years_ogone_noise_reduction.csv', 3, savefig=True)
# kmeans_high_level('./data/2years_ogone_noise_reduction.csv', 3,  percent_only=True, with_date=True)
# isolation_high_level('./data/2years_ogone_noise_reduction.csv', 10, percent_only=False, savefig=False, with_date=True)

### Anomalies searching manually ###
# data = load_data_high_level('./data/2years_az_noise_reduction.csv')
# plot_score_for_finding_problems(data)
# exit()

### Testing algorithms on annoted datasets ###
# modify _ogone_ to _az_ for the az dataset, in both the data and the anomalies
data = load_data_high_level('./data/2years_ogone_noise_reduction.csv')
anomalies = load_anomalies_nicely('./data/anomalies_ogone.csv')

# realize a crossed validation on the kmeans and the isolation forest
### K-means ###
# for i in range(5):
#     kfold_cross_validation(data, anomalies, 10, 'K-means', kmeans_high_level, 3)

### Isolation Forest ###
# for i in range(5):
#     kfold_cross_validation(data, anomalies, 10, 'Isolation Forest', isolation_high_level, 10, random_state=None)

# exit()

#### Test du score pour vérifier qu'il fonctionne correctement ####
# empty_pd = pd.DataFrame()
# random_pd = data[(data.index.hour >= 1) & (data.index.hour <= 6)][:len(anomalies)]
# print(type(random_pd))
# print(random_pd.head())
# random_pd = random_pd.rename(columns={'status': 'predictions'})

# one_random = random_pd.sample(n=4)

# print("100% trouvé - score =", compute_score(anomalies, anomalies), "\n")
# print("0% trouvé - score = ", compute_score(anomalies, empty_pd), "\n")
# print("Test ", compute_score(anomalies, pd.concat([one_random, anomalies])), "\n")


# contamination = 0.001
# for i in range(10):
#     data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)
#     results = isolation_high_level(data_train, data_test, scalers, 50, contamination=contamination, random_state=None, plot_data=False)
#     print(contamination, compute_score(anomalies, results))
#     contamination += 0.003


# load the model from the pickle file
# with open('isolation_forest/high_level/50_trees/model.pkl', 'rb') as f:
#     isolation_forest = pickle.load(f)

# data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)
# data_test['predictions'] = isolation_forest.predict(data_test)

# score = compute_score(anomalies, data_test)
# print(score)

data_train, data_test, scalers = prepare_data_high_level(data, percent_only=True, with_date=False)
# convert back the percentage to their original value
data_test['percentage'] = scalers[0].inverse_transform(data_test['percentage'].values.reshape(-1, 1))
threshold_model(data_test, anomalies)


