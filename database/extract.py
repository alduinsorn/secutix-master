import os

import cx_Oracle
# cx_Oracle.init_oracle_client(lib_dir="/opt/oracle/instantclient_21_12")


class HighLevelData():
    def __init__(self, success_rate=0.0, transaction_count_success=0, total_amount_success=0.0, transaction_count_refused=0, total_amount_refused=0.0, incident=False):
        self.success_rate = success_rate
        self.transaction_count_success = transaction_count_success
        self.total_amount_success = total_amount_success
        self.transaction_count_refused = transaction_count_refused
        self.total_amount_refused = total_amount_refused
        self.incident = incident

    def _convert_to_csv(self):
        return f"{self.success_rate},{self.transaction_count_success},{self.total_amount_success},{self.transaction_count_refused},{self.total_amount_refused},{self.incident}\n"


def request_data(date_start, date_end):
    query = f"""
    SELECT 
        CASE
            WHEN PAYMENT_STATE IN ('REFUSED', 'UNPAID') THEN 'REFUSED'
            WHEN PAYMENT_STATE IN ('PAID', 'SUBMITTED') THEN 'SUCCESS'
            WHEN PAYMENT_STATE IN ('CANC_ADMIN', 'ABANDONED', '3DS_LOST', 'ERROR', 'REFUNDED', 'INPUT', 'REDIRECTED', 'ALIAS_UPD', 'CHOOSING', 'ARCHIVING', 'AUTHO_CONF', 'AUTHORIZED') THEN 'OTHERS'
            ELSE DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE)
        END as status, 
        count(*) AS NB_TRANSACTION, 
        SUM(ORDER_AMOUNT) AS TOTAL_AMOUNT, 
        RATIO_TO_REPORT(COUNT(1)) OVER() * 100 as ratio
    FROM PAYMENT
    WHERE  1=1
        AND creation_date > TO_DATE ('{date_start}', 'YYYY-MM-DD HH24:MI:SS')
        AND creation_date < TO_DATE ('{date_end}', 'YYYY-MM-DD HH24:MI:SS')
    GROUP BY
        CASE
            WHEN PAYMENT_STATE IN ('REFUSED', 'UNPAID') THEN 'REFUSED'
            WHEN PAYMENT_STATE IN ('PAID', 'SUBMITTED') THEN 'SUCCESS'
            WHEN PAYMENT_STATE IN ('CANC_ADMIN', 'ABANDONED', '3DS_LOST', 'ERROR', 'REFUNDED', 'INPUT', 'REDIRECTED', 'ALIAS_UPD', 'CHOOSING', 'ARCHIVING', 'AUTHO_CONF', 'AUTHORIZED') THEN 'OTHERS'
            ELSE DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE)
        END
    """

    cursor.execute(query)
    highLevel = HighLevelData()

    for row in cursor.fetchall():
        # separate the tuple into variables -> the data are like that (status, nb_transaction, total_amount, ratio)
        status, nb_transaction, total_amount, ratio = row
        if status == 'SUCCESS':
            highLevel.success_rate = ratio
            highLevel.transaction_count_success = nb_transaction
            highLevel.total_amount_success = total_amount
        elif status == 'REFUSED':
            highLevel.transaction_count_refused = nb_transaction
            highLevel.total_amount_refused = total_amount
    
    return highLevel


username = "epc4"
password = "epc4"
host_ip = "172.17.0.2"
port = 1521
service_name = "XEPDB1"

try:
    dsn = cx_Oracle.makedsn(host_ip, port, service_name=service_name)
    connection = cx_Oracle.connect(username, password, dsn)
except cx_Oracle.Error as error:
    print(f"Error connecting to the database: {error}")
    exit(1)


cursor = connection.cursor()

data = []
# make a loop to request data of the every 30 minutes of each day for the month of January
for i in range(1, 32):
    for j in range(0, 23):
        # loop between 00 and 30, then 30 and 00 next hour
        if j % 2 == 0:
            date_start = f"2021-01-{i} {j}:00:00"
            date_end = f"2021-01-{i} {j}:30:00"
        else:
            date_start = f"2021-01-{i} {j}:30:00"
            date_end = f"2021-01-{i} {j+1}:00:00"

        data.append(request_data(date_start, date_end))


# write the data into a csv file
with open('data.csv', 'w') as f:
    for highLevel in data:
        f.write(highLevel._convert_to_csv())

connection.commit()
cursor.close()
connection.close()