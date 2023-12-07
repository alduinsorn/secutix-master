import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime

from sklearn.ensemble import IsolationForest


def percent_analysis(unpaid, paid):
    print(f"{(unpaid / (unpaid + paid) * 100):.2f}% ({unpaid+paid})")

def prepare_data(data, onehot_state=False):
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


    norm_data = prepare_data(data, onehot_state)
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
            if savefig: plt.savefig(f'{folder_name}/kmeans_{start_datetime.strftime("%dth_%Hh")}_january_2023_cluster_{i}_with_payment_state.png')
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
            if savefig: plt.savefig(f'{folder_name}/kmeans_{start_datetime.strftime("%dth_%Hh")}_january_2023_cluster_{i}.png')
            plt.show()
    


    # # # plot some of the data points
    # plt.scatter(display_data['CREATION_DATE'], display_data['PAYMENT_STATE'], c=display_data_labels, cmap='rainbow')
    # plt.xlabel('Payment State')
    # plt.ylabel('Amount')
    # plt.show()

def kmeans_high_level():
    pass


def isolation_level_transactions(fname, start_datetime, end_datetime, onehot_state=False, savefig=False):
    data = pd.read_csv(fname)
    data['CREATION_DATE'] = pd.to_datetime(data['CREATION_DATE'])
    
    # display the number of values during the incident period
    # print(len(data[(data['CREATION_DATE'] > start_datetime) & (data['CREATION_DATE'] < end_datetime)]))
    # print(len(data))
    # exit()

    norm_data = prepare_data(data, onehot_state)

    # print(norm_data.columns)
    # exit()

    norm_data['CREATION_DATE'] = pd.to_datetime(norm_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

    display_data = norm_data[(norm_data['CREATION_DATE'] > start_datetime) & (norm_data['CREATION_DATE'] < end_datetime)]
    display_data = display_data.drop(['CREATION_DATE'], axis=1)
    print("display_data", display_data.head())

    ### Contamination value resolved by : 70% of the data of the incident period are anomaly so -> (0.7*1924) / 567437 = 0.00237 -> give no anomaly
    isolation_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.237, random_state=42)

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

        # recreate the payment_method fields for the plot
    
    display_data['PAYMENT_METHOD'] = ''
    for index, row in display_data.iterrows():
        for column in display_data.columns:
            if row[column] == 1 and column.startswith('PAYMENT_METHOD'):
                # remove PAYMENT_METHOD from the name
                display_data.at[index, 'PAYMENT_METHOD'] = column.replace('PAYMENT_METHOD_', '')
                break

    folder_name = f'isolation_forest/{"onehot_payment_state" if onehot_state else "binary_payment_state"}'

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


savefig = True

isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', incident_start_datetime, incident_end_datetime, onehot_state=True, savefig=savefig)
input()
isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', normal_start_datetime, normal_end_datetime, onehot_state=True, savefig=savefig)
input()

isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', incident_start_datetime, incident_end_datetime, onehot_state=False, savefig=savefig)
input()
isolation_level_transactions('./data/NEW_PAYMENT_202311171101_alldata.csv', normal_start_datetime, normal_end_datetime, onehot_state=False, savefig=savefig)
input()

