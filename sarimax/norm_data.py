import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np


def load_data(fname, timestamp=True):
    data = pd.read_csv(fname)
    # check if their is a column unnamed that contains the numeric index
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    if timestamp:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    return data

def transform_raw_data_into_right_type_and_attributs(data):
    # drop the 2023-11-29 because not complete
    data = data.loc[:'2023-11-28']

    data = data[data['state'] != 'ALIAS_OK']
    data = data[data['state'] != 'WAIT_AUTHO']
    data = data[data['state'] != 'initial']
    data.loc[data['state'] == 'AUTHORIZED', 'state'] = 'PAID'
    data.loc[data['state'] == 'SUBMITTING', 'state'] = 'PAID'
    data.loc[data['state'] == 'SUBMI_MANU', 'state'] = 'PAID'
    data.loc[data['state'] == 'SUBMI_TIMO', 'state'] = 'PAID'
    data.loc[data['state'] == 'CANC_ADMIN', 'state'] = 'UNPAID'
    data.loc[data['state'] == 'CANC_USER', 'state'] = 'UNPAID'

    # for each hour, sum the values if same state
    data = data.groupby([data.index, 'payment_type', 'state']).sum()
    # retransform the index
    data = data.reset_index(level=['payment_type', 'state'])

    # add a classic numeric index but keep the actual in a column named 'timestamp'
    data = data.reset_index()
    data = data.rename(columns={'index': 'timestamp'})

    data_ogone = data[data['payment_type'] == 'OGONE_HID'].copy()
    data_datatrans = data[data['payment_type'] == 'DATRS_HID'].copy()
    data_az = data[data['payment_type'] == 'AZ_JSHD'].copy()

    times_range = pd.date_range(start=data['timestamp'].min(), end=data['timestamp'].max(), freq='H')
    STATES = ['PAID', 'UNPAID', 'ABANDONED']
    
    for time in times_range:
        # print(data_datatrans[data_datatrans['timestamp'] == time])

        # verify if the time is in the column 'timestamp'
        if time not in data_datatrans['timestamp'].values: # append new entries to the dataframe without using the timestamp 
            # print(f"{time} is missing")
            data_datatrans = data_datatrans.append({'timestamp': time, 'payment_type': 'DATRS_HID', 'state': 'PAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_datatrans = data_datatrans.append({'timestamp': time, 'payment_type': 'DATRS_HID', 'state': 'UNPAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_datatrans = data_datatrans.append({'timestamp': time, 'payment_type': 'DATRS_HID', 'state': 'ABANDONED', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
        else:
            # get the line with the time and export the state of each line into a list
            lines = data_datatrans[data_datatrans['timestamp'] == time]
            if type(lines) == pd.Series: # if only one line
                lines = pd.DataFrame([lines])
            states = list(lines['state'].values) # get the state of each line
            total_transaction_count_datatrans = data_datatrans[data_datatrans['timestamp'] == time]['transaction_count'].sum() # get the total transaction count of the time
            
            # print(f"Time: {time} - States: {states} - Total transaction count: {total_transaction_count_datatrans}")

            for s in STATES:
                if s in states: # si le state existe dans la ligne on lui rajoute le total_transaction_count
                    # print(f"State {s} is in states")
                    # add the total_transaction_count to the line
                    data_datatrans.loc[(data_datatrans['timestamp'] == time) & (data_datatrans['state'] == s), 'total_transaction_count'] = total_transaction_count_datatrans
                    # print(data_datatrans[data_datatrans['timestamp'] == time])

                else: # sinon on rajoute une ligne avec le state et le total_transaction_count
                    # print(f"State {s} is not in states")
                    data_datatrans = data_datatrans.append({'timestamp': time, 'payment_type': 'DATRS_HID', 'state': s, 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': total_transaction_count_datatrans}, ignore_index=True)
                    # print(data_datatrans[data_datatrans['timestamp'] == time])


        # print(data_datatrans[data_datatrans['timestamp'] == time])
        # input()

        if time not in data_az['timestamp'].values:
            # print(f"{time} is missing")
            data_az = data_az.append({'timestamp': time, 'payment_type': 'AZ_JSHD', 'state': 'PAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_az = data_az.append({'timestamp': time, 'payment_type': 'AZ_JSHD', 'state': 'UNPAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_az = data_az.append({'timestamp': time, 'payment_type': 'AZ_JSHD', 'state': 'ABANDONED', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
        else:
            lines = data_az[data_az['timestamp'] == time]
            if type(lines) == pd.Series:
                lines = pd.DataFrame([lines])
            states = list(lines['state'].values)
            total_transaction_count_az = data_az[data_az['timestamp'] == time]['transaction_count'].sum()

            for s in STATES:
                if s in states:
                    data_az.loc[(data_az['timestamp'] == time) & (data_az['state'] == s), 'total_transaction_count'] = total_transaction_count_az
                else:
                    data_az = data_az.append({'timestamp': time, 'payment_type': 'AZ_JSHD', 'state': s, 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': total_transaction_count_az}, ignore_index=True)

        if time not in data_ogone['timestamp'].values:
            # print(f"{time} is missing")
            data_ogone = data_ogone.append({'timestamp': time, 'payment_type': 'OGONE_HID', 'state': 'PAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_ogone = data_ogone.append({'timestamp': time, 'payment_type': 'OGONE_HID', 'state': 'UNPAID', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
            data_ogone = data_ogone.append({'timestamp': time, 'payment_type': 'OGONE_HID', 'state': 'ABANDONED', 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': 0}, ignore_index=True)
        else:
            lines = data_ogone[data_ogone['timestamp'] == time]
            if type(lines) == pd.Series:
                lines = pd.DataFrame([lines])
            states = list(lines['state'].values)
            total_transaction_count_ogone = data_ogone[data_ogone['timestamp'] == time]['transaction_count'].sum()

            for s in STATES:
                if s in states:
                    data_ogone.loc[(data_ogone['timestamp'] == time) & (data_ogone['state'] == s), 'total_transaction_count'] = total_transaction_count_ogone
                else:
                    data_ogone = data_ogone.append({'timestamp': time, 'payment_type': 'OGONE_HID', 'state': s, 'transaction_count': 0, 'amount_count': 0.0, 'percentage': 0.0, 'total_transaction_count': total_transaction_count_ogone}, ignore_index=True)


    # sort the index
    data_ogone = data_ogone.sort_index()
    data_datatrans = data_datatrans.sort_index()
    data_az = data_az.sort_index()

    # save data with the index as a column named 'timestamp'
    data_ogone.to_csv('data/2years_ogone.csv', index=True)
    data_datatrans.to_csv('data/2years_datatrans.csv', index=True)
    data_az.to_csv('data/2years_az.csv', index=True) 
    exit()

def display_info_paytype(data, paytype, max_hour, threshold=150):
    spec_data = data[data['payment_type'] == paytype]
    print(f"\nData {paytype}: {len(spec_data)}")
    print(f"Data {paytype} with state PAID: {len(spec_data[spec_data['state'] == 'PAID'])} on {max_hour} ({(len(spec_data[spec_data['state'] == 'PAID']) / max_hour) * 100:.2f}%) possible hours")
    grouped = spec_data.groupby(spec_data.index).sum(numeric_only=True)
    print(f"Mean transaction count: {grouped['transaction_count'].mean():.2f}")
    
    spec_data_thresh = spec_data[spec_data['transaction_count'] > threshold]
    grouped2 = spec_data_thresh.groupby(spec_data_thresh.index).sum(numeric_only=True)
    
    print(f"Data {paytype} after filtering threshold: {len(spec_data)}")
    for i in range(24):
        print(f"Hour {i}: {len(grouped2[grouped2.index.hour == i])}\tmean thresh: {grouped2[grouped2.index.hour == i]['transaction_count'].mean():.2f}\tmean global: {grouped[grouped.index.hour == i]['transaction_count'].mean():.2f}")

def plot_data(data, attribut, paytype, savefig=False, state=None, percentile=False):
    sept_data_paytype = data[data['payment_type'] == paytype]
    figname = f'{paytype}-{attribut}'
    if state: 
        sept_data_paytype = sept_data_paytype[sept_data_paytype['state'] == state]
        figname = f'{paytype}-{attribut}-{state}'
    else:
        sept_data_paytype = sept_data_paytype.groupby(sept_data_paytype.index).sum(numeric_only=True)

    plt.figure(figsize=(20, 10))
    plt.plot(sept_data_paytype[attribut])

    if percentile:
        plt.axhline(y=sept_data_paytype[attribut].quantile(0.25), color='r', linestyle='--', label=f'25th percentile ({sept_data_paytype[attribut].quantile(0.25):.2f})')
        plt.axhline(y=sept_data_paytype[attribut].quantile(0.33), color='orange', linestyle='--', label=f'33th percentile ({sept_data_paytype[attribut].quantile(0.33):.2f})')
        plt.axhline(y=sept_data_paytype[attribut].quantile(0.50), color='g', linestyle='--', label=f'50th percentile ({sept_data_paytype[attribut].quantile(0.50):.2f})')
        plt.axhline(y=sept_data_paytype[attribut].quantile(0.75), color='purple', linestyle='--', label=f'75th percentile ({sept_data_paytype[attribut].quantile(0.75):.2f})')
        # display the mean with a dotted line
        plt.axhline(y=sept_data_paytype[attribut].mean(), color='b', linestyle='--', label=f'mean ({sept_data_paytype[attribut].mean():.2f})')
        plt.legend()
        figname = f'{figname}-percentile'

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.title(f'{paytype} - {attribut}')
    plt.autoscale()
    if savefig: plt.savefig(f'{figname}.png')
    plt.show()

def data_analysis(fname='./data/az+ogone+datatrans_2021_12_01_to_2023_28_11.csv'):
    data = load_data(fname)
    # get min and max date
    min_date = data.index.min()
    max_date = data.index.max()
    # count the number of hour between min and max date
    nb_hour = (max_date - min_date).days * 24 + (max_date - min_date).seconds // 3600
    print(f"\nnb_hour(nb_line): {nb_hour}\n")

    # display_info_paytype(data, 'OGONE_HID', nb_hour)
    # display_info_paytype(data, 'DATRS_HID', nb_hour)
    # display_info_paytype(data, 'AZ_JSHD', nb_hour)

    sept_data = data['2023-09-01':'2023-09-30']

    attribut = 'percentage'

    plot_data(sept_data, attribut, 'OGONE_HID', savefig=True, state='PAID', percentile=False)
    plot_data(sept_data, attribut, 'DATRS_HID', savefig=True, state='PAID', percentile=False)
    plot_data(sept_data, attribut, 'AZ_JSHD', savefig=True, state='PAID', percentile=False)



### Partir sur 50 transactions par heure pour être considéré comme données utiles 
### Si pas assez voir si on drop l'heure ou meme toute la journée -> de toute facon pas relevant (pas de gros impact "économique")

def display_info(fname, threshold):
    data = load_data(fname)
    print(data.head())
    # keep only the paid state
    data = data[data['state'] == 'PAID']
    # keep only some attributs: timestamp,percentage,total_transaction_count
    data = data[['percentage', 'total_transaction_count']]

    # count the number of hour between min and max date 
    min_date = data.index.min()
    max_date = data.index.max()
    print(f"min date: {min_date}")
    print(f"max date: {max_date}")

    nb_hour = (max_date - min_date).days * 24 + (max_date - min_date).seconds // 3600 + 1 # +1 because we count the first hour
    print(f"number of hour between min and max date: {nb_hour} - equal to {nb_hour / 24} days or elements for each hour")
    
    # search the time that is missing between min and max date
    times_range = pd.date_range(start=min_date, end=max_date, freq='H')
    for time in times_range:
        # check if the time is in the index
        if time not in data.index:
            print(f"{time} is missing")

    # print mean of the total_transaction_count and the 75th percentile
    print(f"mean of the total_transaction_count: {data['total_transaction_count'].mean():.2f}")
    print(f"75th percentile of the total_transaction_count: {data['total_transaction_count'].quantile(0.75):.2f}")

    confident_data_by_hour = []
    for i in range(24):
        print(f"Hour {i}: {len(data[(data.index.hour == i) & (data['total_transaction_count'] > threshold)])}")
        confident_data_by_hour.append(len(data[(data.index.hour == i) & (data['total_transaction_count'] > threshold)]))

    plt.figure(figsize=(20, 10))
    plt.bar(range(24), confident_data_by_hour)
    plt.title(f"Number of confident data by hour with a threshold of {threshold}")
    plt.xlabel("Hour")
    plt.ylabel("Number of confident data")
    plt.xticks(range(24))
    plt.ylim(0, nb_hour / 24 + 1)
    plt.savefig(f"confident_data_by_hour_{fname.split('_')[1].split('.')[0]}_{threshold}.png")
    plt.show()

def data_noise_reduction_analysis(fname, threshold, morning_hours=(1, 6), confident_hours=(7, 23), include_0=True):
    data = load_data(fname)
    print(data.head(1)['payment_type'])
    data = data[data['state'] == 'PAID']
    data = data[['percentage', 'total_transaction_count']]

    enough_data = data[data['total_transaction_count'] > threshold]

    # print the mean of the 'percentage' for the hours between 1 and 6 (included)
    print(f"mean of the percentage between {morning_hours[0]}h and {morning_hours[1]}h: \t\t\t{data['percentage'][(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])].mean():.2f} (std: {data['percentage'][(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])].std():.2f}) - number of data: {len(data[(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])])}")
    # print the mean of the 'percentage' for the hours between 7 and 23 (included) include 0 also
    data_confident = data[(data.index.hour >= confident_hours[0]) & (data.index.hour <= confident_hours[1])]
    if include_0: data_confident = pd.concat([data_confident, data[data.index.hour == 0]])
    print(f"mean of the percentage between {confident_hours[0]}h and {confident_hours[1]}h: \t\t\t{data_confident['percentage'].mean():.2f} (std: {data_confident['percentage'].std():.2f}) - number of data: {len(data_confident)}")

    # do the same but with "enough_data"
    print(f"(enough data) mean of the percentage between {morning_hours[0]}h and {morning_hours[1]}h: \t{enough_data['percentage'][(enough_data.index.hour >= morning_hours[0]) & (enough_data.index.hour <= morning_hours[1])].mean():.2f} - number of data: {len(enough_data[(enough_data.index.hour >= morning_hours[0]) & (enough_data.index.hour <= morning_hours[1])])}")
    enough_data_confident = enough_data[(enough_data.index.hour >= confident_hours[0]) & (enough_data.index.hour <= confident_hours[1])]
    if include_0: enough_data_confident = pd.concat([enough_data_confident, enough_data[enough_data.index.hour == 0]])
    print(f"(enough data) mean of the percentage between {confident_hours[0]}h and {confident_hours[1]}h: \t{enough_data_confident['percentage'].mean():.2f} - number of data: {len(enough_data_confident)}")
    print(f"max of the percentage between {confident_hours[0]}h and {confident_hours[1]}h: \t\t\t{enough_data_confident['percentage'].quantile(0.75):.2f}")
    print()

    # do the same but this time using the 'total_transaction_count' instead of the 'percentage'
    print(f"mean of the total_transaction_count between {morning_hours[0]}h and {morning_hours[1]}h: \t\t{data['total_transaction_count'][(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])].mean():.2f} (std: {data['total_transaction_count'][(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])].std():.2f}) - number of data: {len(data[(data.index.hour >= morning_hours[0]) & (data.index.hour <= morning_hours[1])])}")
    print(f"mean of the total_transaction_count between {confident_hours[0]}h and {confident_hours[1]}h: \t\t{data_confident['total_transaction_count'].mean():.2f} (std: {data_confident['total_transaction_count'].std():.2f}) - number of data: {len(data_confident)}")

    print(f"(enough data) mean of the total_transaction_count between {morning_hours[0]}h and {morning_hours[1]}h: \t{enough_data['total_transaction_count'][(enough_data.index.hour >= morning_hours[0]) & (enough_data.index.hour <= morning_hours[1])].mean():.2f} - number of data: {len(enough_data[(enough_data.index.hour >= morning_hours[0]) & (enough_data.index.hour <= morning_hours[1])])}")
    print(f"(enough data) mean of the total_transaction_count between {confident_hours[0]}h and {confident_hours[1]}h: \t{enough_data_confident['total_transaction_count'].mean():.2f} - number of data: {len(enough_data_confident)}")


def data_noise_reduction_percentile_analysis(fname, threshold):
    data = load_data(fname)
    data = data[data['state'] == 'PAID']
    data = data[['percentage', 'total_transaction_count']]
    # remove data that have a percentage = 0
    data = data[data['percentage'] != 0]

    confident_data = data[data['total_transaction_count'] > threshold]

    percentile = 0.60

    # get the 75th that better represent the data
    percentile75th_percentage = data['percentage'].quantile(percentile)
    percentile75th_total_transaction_count = data['total_transaction_count'].quantile(percentile)
    confident_percentile75th_percentage = confident_data['percentage'].quantile(percentile)
    confident_percentile75th_total_transaction_count = confident_data['total_transaction_count'].quantile(percentile)
    # get the std of the data in general
    confident_std_percentage = confident_data['percentage'].std()
    confident_std_total_transaction_count = confident_data['total_transaction_count'].std()


    print(f"{percentile * 100:.0f}th percentile of the percentage: {percentile75th_percentage:.2f}")
    print(f"{percentile * 100:.0f}th percentile of the percentage with a threshold of {threshold}: {confident_percentile75th_percentage:.2f}")
    print(f"{percentile * 100:.0f}th percentile of the total_transaction_count: {percentile75th_total_transaction_count:.2f}")
    print(f"{percentile * 100:.0f}th percentile of the total_transaction_count with a threshold of {threshold}: {confident_percentile75th_total_transaction_count:.2f}")
    print()

    middle_percentage = (percentile75th_percentage + confident_percentile75th_percentage) / 2
    middle_total_transaction_count = (percentile75th_total_transaction_count + confident_percentile75th_total_transaction_count) / 2
    middle_std_percentage = (confident_std_percentage) / 4
    middle_std_total_transaction_count = (confident_std_total_transaction_count) / 4

    print(f"middle of the percentage: {middle_percentage:.2f} (std: {middle_std_percentage:.2f})")
    print(f"middle of the total_transaction_count: {middle_total_transaction_count:.2f} (std: {middle_std_total_transaction_count:.2f})")
    print()

    print(f"mean of the percentage: {data['percentage'].mean():.2f} (std: {data['percentage'].std():.2f})")
    print(f"mean of the percentage with a threshold of {threshold}: {confident_data['percentage'].mean():.2f} (std: {confident_data['percentage'].std():.2f})")
    print(f"mean of the total_transaction_count: {data['total_transaction_count'].mean():.2f} (std: {data['total_transaction_count'].std():.2f})")
    print(f"mean of the total_transaction_count with a threshold of {threshold}: {confident_data['total_transaction_count'].mean():.2f} (std: {confident_data['total_transaction_count'].std():.2f})")
    print()

def data_noise_reduction(fname, threshold, mean_total_transaction_count):
    data = load_data(fname)
    data = data[data['state'] == 'PAID']
    data = data[['percentage', 'total_transaction_count']]
    # sort the data by the timestamp
    data = data.sort_index()

    # sept_data = data['2023-09-01':'2023-09-30']
    # print(sept_data[sept_data['percentage'] < 30])


    normal_distribution_percentage = np.random.normal(75, 3.5, len(data[data['total_transaction_count'] < threshold]))
    data.loc[data['total_transaction_count'] < threshold, 'percentage'] = normal_distribution_percentage

    # generate a normal distribution around the mean_total_transaction_count and should not go under the threshold
    normal_distribution_total_transaction_count = np.random.normal(mean_total_transaction_count, 25, len(data[data['total_transaction_count'] < threshold]))
    normal_distribution_total_transaction_count[normal_distribution_total_transaction_count < threshold] = threshold
    normal_distribution_total_transaction_count = normal_distribution_total_transaction_count.astype(int)

    data.loc[data['total_transaction_count'] < threshold, 'total_transaction_count'] = normal_distribution_total_transaction_count

    print(data[(data.index.hour >= 7) & (data.index.hour <= 23) & (data['total_transaction_count'] < 2*threshold)].head(10))

    # create a plot for the september month
    # sept_data = data['2023-09-01':'2023-09-30']
    # print all the data in the console without truncation
    # pd.set_option('display.max_rows', None)
    # print(sept_data)

    # print(sept_data[sept_data['percentage'] < 30])

    # plt.figure(figsize=(20, 10))
    # plt.plot(sept_data['percentage'])
    # plt.show()

    # change the percentage column to have only 1 decimal
    data['percentage'] = data['percentage'].round(1)
    
    data = data.sort_index()
    data = data.reset_index()
    data = data.rename(columns={'index': 'timestamp'})
    # save the file
    data.to_csv(f'{fname.split(".")[0]}_noise_reduction.csv', index=False)

# transform_raw_data_into_right_type_and_attributs(load_data('data/az+ogone+datatrans_2021_12_01_to_2023_28_11.csv'))


# data_analysis()

# display_info('data/2years_ogone.csv', 150)
# display_info('data/2years_datatrans.csv', 50)
# display_info('data/2years_az.csv', 50)

# data_noise_reduction_analysis('data/2years_ogone.csv', 150, (2, 6), (7, 23), include_0=True)
# data_noise_reduction_analysis('data/2years_datatrans.csv', 50, (1, 7), (8, 23), include_0=True)
# data_noise_reduction_analysis('data/2years_az.csv', 50, (0, 7), (8, 23), include_0=False)


# data_noise_reduction_percentile_analysis('data/2years_ogone.csv', 150)
# data_noise_reduction_percentile_analysis('data/2years_datatrans.csv', 50)
# data_noise_reduction_percentile_analysis('data/2years_az.csv', 50)


# data = load_data('data/2years_datatrans.csv')
# data = data[data['state'] == 'PAID']
# data = data[['percentage', 'total_transaction_count']]

# print(data[data['percentage'] < 50])

data_noise_reduction('data/2years_ogone.csv', 150, 90)
data_noise_reduction('data/2years_datatrans.csv', 50, 75)
data_noise_reduction('data/2years_az.csv', 50, 75)