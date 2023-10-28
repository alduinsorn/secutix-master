# file to load incidents from PSP and add a column to the dataframe when there is an incident
# in addition this file will also combine some columns to create a new one and discard others


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Union
from enum import Enum

class Incident:
    def __init__(self, title: str = "Unknown", payment_methods: List[str] = [], services: List[str] = [], identified_datetime: List[str] = [], resolved_datetime: List[str] = [], raw: str = ""):

        self.title = title
        self.payment_methods = payment_methods
        self.services = services
        self.identified_datetime = identified_datetime
        self.resolved_datetime = resolved_datetime
        self.raw = raw
        # self.card_datetime = card_datetime
        # self.incident_type = incident_type

    def _to_dict_all(self) -> dict:
        incident_dict = {
            "title": self.title,
            "payment_methods": self.payment_methods,
            "services": self.services,
            "identified_datetime": self.identified_datetime,
            "resolved_datetime": self.resolved_datetime,
            "raw": self.raw,
            # "card_datetime": str(self.card_datetime)
        }
        # if self.incident_type:
        #     incident_dict["incident_type"] = {
        #         "id" : self.incident_type.value,
        #         "title" : self.incident_type.title
        #     }
        return incident_dict

    def _to_dict(self) -> dict:
        # compute severity or do it when we have all the incidents ?
        # need to separate the services into payment method and service ?

        incident_dict = {
            "title": str(self.title),
            "payment_methods": self.payment_methods,
            "services": self.services,
            "identified_datetime": str(self.identified_datetime),
            "resolved_datetime": str(self.resolved_datetime),
            "raw": self.raw,
        }
        return incident_dict
    
    
    def __str__(self) -> str:
        attributes: dict = self._to_dict()
        json_str: str = "{\n"
        for key, value in attributes.items():
            json_str += f'    "{key}": "{value}",\n'
        json_str += "}"
        return json_str


# useless, done during the database extraction
def convert_general_data_to_real(fn_data):
    fn_data = './data/data_ogone.csv'
    data = pd.read_csv(fn_data)
    # first part
    ## every line is an hour, addition the value of the paid_transaction_count, unpaid_transaction_count and abandoned_transaction_count to get the total_transaction_count
    ## then we drop the 3 columns and rename the total_transaction_count to transaction_count
    data['transaction_count'] = data['paid_transaction_count'] + data['unpaid_transaction_count'] + data['abandoned_transaction_count']
    data = data.drop(columns=['paid_transaction_count', 'unpaid_transaction_count', 'abandoned_transaction_count'])
    ## drop the paid_total_amount, unpaid_total_amount and abandoned_total_amount columns
    data = data.drop(columns=['paid_total_amount', 'unpaid_total_amount', 'abandoned_total_amount'])
    ## drop the unpaid_rate and abandoned_rate columns
    data = data.drop(columns=['unpaid_rate', 'abandoned_rate'])
    ## create a new column at the beginning of the dataframe called index
    data['index'] = np.arange(len(data))
    ## set the index of the dataframe to the index column
    data = data.set_index('index')
    ## save the data in a new file called real_data_ogone.csv
    data.to_csv('../database/data/real_data_ogone.csv', index=True)




fn_data = './data/real_data_ogone.csv'
data = pd.read_csv(fn_data)

fn_incidents = '../scraping/data/ogone_incidents.json'
incidents = pd.read_json(fn_incidents)

print(incidents[2023])


# add a column 'incident' that is a boolean, the default is False
data['incident'] = False

# create an array that contains the month of the year
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september']

for month in months:
    month_incidents = incidents[2023][month]
    print(f"Month: {month} - number of incidents: {len(month_incidents)}")
    count = 0
    for incident in month_incidents:
        # incident_datetime = datetime.strptime(incident['identified_datetime'], '%Y-%m-%d %H:%M')
        # incident_datetime = incident_datetime.replace(minute=0, second=0)
        # incident_datetime = incident_datetime.strftime('%Y-%m-%d %H:%M:%S')
        # data.loc[data[data['timestamp'] == incident_datetime].index.values, 'incident'] = True

        incident_datetime_start = datetime.strptime(incident['identified_datetime'], '%Y-%m-%d %H:%M')
        incident_datetime_end = datetime.strptime(incident['resolved_datetime'], '%Y-%m-%d %H:%M')

        # compute the difference between the start and the end of the incident
        diff = incident_datetime_end - incident_datetime_start
        # if the difference is less than 5 minutes, skip the incident
        if diff.seconds < 300:
            count += 1
            continue

        # get the title, if the title contains 'scheduled', 'maintenance' or 'slight', skip the incident
        title = incident['title'].lower()
        if 'scheduled' in title or 'maintenance' in title or 'slight' in title:
            count += 1
            continue

        # now we have a real incident, we can add it to the dataframe
        incident_datetime_start = incident_datetime_start.replace(minute=0, second=0)
        incident_datetime_start = incident_datetime_start.strftime('%Y-%m-%d %H:%M:%S')
        data.loc[data[data['timestamp'] == incident_datetime_start].index.values, 'incident'] = True
        

        plain_hour_end = incident_datetime_end.replace(minute=0, second=0)
        plain_diff = incident_datetime_end - plain_hour_end
        if plain_diff.seconds > 300:
            print(f"{plain_diff.seconds} > 300, incident_datetime_end: {incident_datetime_end}")

            incident_datetime_end = incident_datetime_end.replace(minute=0, second=0)
            incident_datetime_end = incident_datetime_end.strftime('%Y-%m-%d %H:%M:%S')
            data.loc[data[data['timestamp'] == incident_datetime_end].index.values, 'incident'] = True
    
    print(f"Number of incidents skipped: {count}/{len(month_incidents)}")

# display the dataframe where there is an incident
print(data[data['incident'] == True])
# print the number of incidents in the dataframe
print(len(data[data['incident'] == True]))


# drop the timestamp column
data = data.drop(columns=['timestamp'])
# add an index column
data['index'] = np.arange(len(data))
# set the index of the dataframe to the index column
data = data.set_index('index')
# save the dataframe in a new file called real_data_ogone_incidents.csv
data.to_csv('./data/real_data_ogone_incidents.csv', index=True)
