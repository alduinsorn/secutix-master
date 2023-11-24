import pandas as pd
from datetime import datetime
from datetime import timedelta
import time as time_lib
import os
import threading

def compute_time_range(data, time_list, thread_id):
    th25 = len(time_list) // 4
    th50 = len(time_list) // 2
    th75 = th25 * 3

    for index, time in enumerate(time_list):
        if index == th25:
            print(f"cpu {thread_id}: 25% done")
        elif index == th50:
            print(f"cpu {thread_id}: 50% done")
        elif index == th75:
            print(f"cpu {thread_id}: 75% done")


        # start_time = time_lib.time()
        data_hour = data[data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')]
        # print(data_hour)
        total_transaction_count = data_hour['TRANSACTION_COUNT'].sum()
        # print(f"total_transaction_count: {total_transaction_count}")

        unique_payment_types = data_hour['PAYMENT_TYPE'].unique().tolist()
        if 'UNKNOWN' in unique_payment_types:
            unique_payment_types.remove('UNKNOWN')

        # ce nombre de transaction doit être réparti sur les différents payment types
        count_payment_type_unknown = data_hour[data_hour['PAYMENT_TYPE'] == 'UNKNOWN']['TRANSACTION_COUNT'].sum()
        # print(f"count_payment_type_unknown: {count_payment_type_unknown}")

        for payment_type in unique_payment_types: # OGONE_HID, DATRS_HID, etc...
            # print(data_hour[data_hour['PAYMENT_TYPE'] == payment_type])

            unique_payment_methods = data_hour[data_hour['PAYMENT_TYPE'] == payment_type]['PAYMENT_METHOD'].unique().tolist()
            if 'UNKNOWN' in unique_payment_methods:
                unique_payment_methods.remove('UNKNOWN')

            # utilisé pour la répartition des transactions avec payment_method = UNKNOWN
            count_transaction_specific_payment_type = data_hour[data_hour['PAYMENT_TYPE'] == payment_type]['TRANSACTION_COUNT'].sum()
            # print(f"count_transaction_specific_payment_type before: {count_transaction_specific_payment_type}")
            # ajout a ce nombre de transaction, le nombre de transaction avec payment_type = UNKNOWN
            count_transaction_specific_payment_type += round(count_payment_type_unknown * (count_transaction_specific_payment_type / total_transaction_count))
            # print(f"count_transaction_specific_payment_type after: {count_transaction_specific_payment_type}")

            # On va devoir distribuer ces transactions sur les différents payment methods, suivant ce que représente chaque payment method
            count_unknown = data_hour[(data_hour['PAYMENT_TYPE'] == payment_type) & (data_hour['PAYMENT_METHOD'] == 'UNKNOWN')]['TRANSACTION_COUNT'].sum()

            for payment_method in unique_payment_methods: # VISA, MASTERCARD, AMEX, etc...
                # pour chaque payment method, on doit trouver PAID, UNPAID et ABANDONED. Si il manque un de ces 3, on le met à 0
                try:
                    count_paid = data_hour[(data_hour['PAYMENT_TYPE'] == payment_type) & (data_hour['PAYMENT_METHOD'] == payment_method) & (data_hour['STATE'] == 'PAID')]['TRANSACTION_COUNT'].sum()
                except:
                    count_paid = 0
                try:
                    count_unpaid = data_hour[(data_hour['PAYMENT_TYPE'] == payment_type) & (data_hour['PAYMENT_METHOD'] == payment_method) & (data_hour['STATE'] == 'UNPAID')]['TRANSACTION_COUNT'].sum()
                except:
                    count_unpaid = 0
                try:
                    count_abandoned = data_hour[(data_hour['PAYMENT_TYPE'] == payment_type) & (data_hour['PAYMENT_METHOD'] == payment_method) & (data_hour['STATE'] == 'ABANDONED')]['TRANSACTION_COUNT'].sum()
                except:
                    count_abandoned = 0

                count_refused = count_unpaid + count_abandoned

                percent_representation = (count_paid + count_refused) / count_transaction_specific_payment_type # not in real percent
                unknown_to_add = round(count_unknown * percent_representation) # if above x.5 go to up, else go to down
                
                # compute the new percentage
                real_percentage_paid = count_paid / (count_paid + count_refused + unknown_to_add) * 100 if (count_paid + count_refused + unknown_to_add) != 0 else 0
                real_percentage_unpaid = count_unpaid / (count_paid + count_refused + unknown_to_add) * 100 if (count_paid + count_refused + unknown_to_add) != 0 else 0
                real_percentage_abandoned = count_abandoned / (count_paid + count_refused + unknown_to_add) * 100 if (count_paid + count_refused + unknown_to_add) != 0 else 0
                # assign the new percentage to the dataframe under the name "REAL_PERCENTAGE", use the generate dataframe called "data"
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'PAID'), 'REAL_PERCENTAGE'] = real_percentage_paid
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'UNPAID'), 'REAL_PERCENTAGE'] = real_percentage_unpaid
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'ABANDONED'), 'REAL_PERCENTAGE'] = real_percentage_abandoned

                # create a new column for the new number of transaction_count for abandoned and unpaid
                percent_representation_unpaid = count_unpaid / count_refused if count_refused != 0 else 0 
                # print(f"percent_representation_unpaid: {percent_representation_unpaid} (count_unpaid: {count_unpaid} / count_refused: {count_refused})")
                real_count_unpaid = round(count_unpaid + (unknown_to_add * percent_representation_unpaid))
                percent_representation_abandoned = count_abandoned / count_refused if count_refused != 0 else 0
                real_count_abandoned = round(count_abandoned + (unknown_to_add * percent_representation_abandoned))

                # assign the new number of transaction_count to the dataframe under the name "REAL_TRANSACTION_COUNT", use the generate dataframe called "data"
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'UNPAID'), 'REAL_TRANSACTION_COUNT'] = real_count_unpaid
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'ABANDONED'), 'REAL_TRANSACTION_COUNT'] = real_count_abandoned
                # doesn't change compared to the original dataframe
                data.loc[(data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['PAYMENT_TYPE'] == payment_type) & (data['PAYMENT_METHOD'] == payment_method) & (data['STATE'] == 'PAID'), 'REAL_TRANSACTION_COUNT'] = count_paid


            # print("---------AFTER----------")
            # display the dataframe "data" after the changes
            # print(data_hour[data_hour['PAYMENT_TYPE'] == payment_type])
        # pd.set_option('display.max_rows', None)
        # print(data[data['DATETIME'] == time.strftime('%Y-%m-%d %H:%M:%S.000')])
        # input()

        # end_time = time_lib.time()
        # minutes, secondes = divmod(end_time - start_time, 60)
        # print(f"Time to process one hour of data: {minutes} minutes and {secondes} secondes")
        # if input("Continue? (y/n): ") != 'y':
        #     exit()

    partial_results.append((thread_id, data))

def create_time_range(start, end):
    time_list = []
    current_time = start
    while current_time <= end:
        time_list.append(current_time)
        current_time = current_time + timedelta(hours=1)
    return time_list

def load_wrong_data(fn='HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise.csv'):
    data = pd.read_csv(fn)
    # if the PAYMENT_METHOD is 'VISA', 'MASTERCARD' or 'AMEX' and that the PAYMENT_METHOD is OGONE_RED, DATRS_RED change it to OGONE_HID, DATRS_HID
    data.loc[(data['PAYMENT_METHOD'] == 'VISA') & (data['PAYMENT_METHOD'] == 'OGONE_RED'), 'PAYMENT_METHOD'] = 'OGONE_HID'
    data.loc[(data['PAYMENT_METHOD'] == 'MASTERCARD') & (data['PAYMENT_METHOD'] == 'OGONE_RED'), 'PAYMENT_METHOD'] = 'OGONE_HID'
    data.loc[(data['PAYMENT_METHOD'] == 'AMEX') & (data['PAYMENT_METHOD'] == 'OGONE_RED'), 'PAYMENT_METHOD'] = 'OGONE_HID'
    data.loc[(data['PAYMENT_METHOD'] == 'VISA') & (data['PAYMENT_METHOD'] == 'DATRS_RED'), 'PAYMENT_METHOD'] = 'DATRS_HID'
    data.loc[(data['PAYMENT_METHOD'] == 'MASTERCARD') & (data['PAYMENT_METHOD'] == 'DATRS_RED'), 'PAYMENT_METHOD'] = 'DATRS_HID'
    data.loc[(data['PAYMENT_METHOD'] == 'AMEX') & (data['PAYMENT_METHOD'] == 'DATRS_RED'), 'PAYMENT_METHOD'] = 'DATRS_HID'

def concatenate_n_export(months):
    files = os.listdir()
    # keep only the files that contains a month in the name
    right_files = []
    for file in files:
        if file.endswith('.csv'):
            if file.split('_')[0] in months:
                right_files.append(file)

    # open every file and then concatenate them into one dataframe
    openend = []
    for file in right_files:
        print(f"Opening {file}...")
        file_data = pd.read_csv(file)
        openend.append(file_data)
        
    new_data = pd.concat(openend)
    # sort the dataframe by datetime
    new_data = new_data.sort_values(by=['DATETIME'])
    # save the dataframe "data" to a csv file
    new_data.to_csv('HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right.csv', index=False)

def compute_by_month(years, months):
    for year in years:
        for mindex, month in enumerate(months):
            print(f"Processing {month} {year}...")
            min_timestamp = datetime.strptime(f'{year}-{mindex+1}-01 00:00:00.000', '%Y-%m-%d %H:%M:%S.000')
            max_timestamp = datetime.strptime(f'{year}-{mindex+1}-{days_in_month[mindex]} 23:00:00.000', '%Y-%m-%d %H:%M:%S.000')

            # verify if the min and max timestamp are in the dataframe
            if min_timestamp < datetime.strptime(data['DATETIME'].min(), '%Y-%m-%d %H:%M:%S.000'):
                continue
            if max_timestamp > datetime.strptime(data['DATETIME'].max(), '%Y-%m-%d %H:%M:%S.000'):
                continue

            # create a list of all the timestamps between min and max, by hour
            time_list = create_time_range(min_timestamp, max_timestamp)

            # print(f"Number of hours range to process: {len(time_list)}")
            # if input("Continue? (y/n): ") != 'y':
            #     exit()


            nb_cpu = os.cpu_count()
            elements_per_cpu = len(time_list) // nb_cpu
            general_time_list = []

            # split the time_list into nb_cpu parts
            for i in range(nb_cpu):
                ab = time_list[i*elements_per_cpu:(i+1)*elements_per_cpu]
                general_time_list.append(ab)
                print(f"cpu {i}: {len(ab)}")
                print(f"first: {ab[0]}")

            # # for testing keep only 2 data
            # for i in range(nb_cpu):
            #     general_time_list[i] = general_time_list[i][:10]

            # create a list of dataframes, one for each cpu
            data_per_cpu = []
            for i in range(nb_cpu):
                min_time = general_time_list[i][0]
                max_time = general_time_list[i][-1]

                the_range_data = data[(data['DATETIME'] >= min_time.strftime('%Y-%m-%d %H:%M:%S.000')) & (data['DATETIME'] <= max_time.strftime('%Y-%m-%d %H:%M:%S.000'))].copy()
                data_per_cpu.append(the_range_data)



            global partial_results
            partial_results = []
            start_time = time_lib.time()

            threads = []
            for i in range(nb_cpu):
                thread = threading.Thread(target=compute_time_range, args=(data_per_cpu[i], general_time_list[i], i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            print(f"Time to process all the data: {time_lib.time() - start_time} secondes")

            final_results = pd.concat([partial_results[i][1] for i in range(nb_cpu)])
            # sort the dataframe by datetime
            final_results = final_results.sort_values(by=['DATETIME'])

            # save the dataframe "data" to a csv file
            # data.to_csv('HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right.csv', index=False)
            # final_results.to_csv('HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right.csv', index=False)
            final_results.to_csv(f'{month}_{year}-HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right.csv', index=False)

def load_new_data(fn='HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right.csv'):
    data = pd.read_csv(fn)
    # data = data.set_index('DATETIME')
    return data

def export_specific(data):
    columns = ['DATETIME', 'PAYMENT_TYPE', 'PAYMENT_METHOD', 'STATE', 'REAL_PERCENTAGE', 'REAL_TRANSACTION_COUNT']
    data = data[columns]
    data = data[data['STATE'] == 'PAID'] # keep only PAID
    data.to_csv('HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right_PAID.csv', index=False)

years = [2022, 2023]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October' ,'November', 'December']
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31 ,30, 31]

data = load_new_data()
export_specific(data)
