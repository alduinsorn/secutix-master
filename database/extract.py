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


class HighLevelData2():
    def __init__(self, paid_rate=0.0, paid_transaction_count=0, paid_total_amount=0.0, unpaid_rate=0.0, unpaid_transaction_count=0, unpaid_total_amount=0.0, abandoned_rate=0.0, abandoned_transaction_count=0, abandoned_total_amount=0.0):
        self.paid_rate = paid_rate
        self.paid_transaction_count = paid_transaction_count
        self.paid_total_amount = paid_total_amount
        self.unpaid_rate = unpaid_rate
        self.unpaid_transaction_count = unpaid_transaction_count
        self.unpaid_total_amount = unpaid_total_amount
        self.abandoned_rate = abandoned_rate
        self.abandoned_transaction_count = abandoned_transaction_count
        self.abandoned_total_amount = abandoned_total_amount
        

    def _convert_to_csv(self):
        return f"{self.paid_rate},{self.paid_transaction_count},{self.paid_total_amount},{self.unpaid_rate},{self.unpaid_transaction_count},{self.unpaid_total_amount},{self.abandoned_rate},{self.abandoned_transaction_count},{self.abandoned_total_amount}\n"


def request_data(date_start, date_end, cursor):
    query = f"""
    SELECT 
        CASE
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('REFUSED', 'UNPAID') THEN 'REFUSED'
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('PAID', 'SUBMITTED', 'REFUNDED') THEN 'PAID'
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('CANC_ADMIN', 'ABANDONED', '3DS_LOST', 'ERROR', 'INPUT', 'REDIRECTED', 'ALIAS_UPD', 'CHOOSING', 'ARCHIVING', 'AUTHO_CONF', 'AUTHORIZED') THEN 'OTHERS'
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
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('REFUSED', 'UNPAID') THEN 'REFUSED'
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('PAID', 'SUBMITTED', 'REFUNDED') THEN 'PAID'
            WHEN DECODE(BITAND(PREVIOUS_STATES, 360460), 
                32768, 'ABANDONED',
                32776, 'ABANDONED',
                32780, '3DS_LOST',
                294912, 'ABANDONED',
                294920, 'ABANDONED',
                294924, '3DS_LOST',
                327692, 'UNPAID',
                PAYMENT_STATE) IN ('CANC_ADMIN', 'ABANDONED', '3DS_LOST', 'ERROR', 'INPUT', 'REDIRECTED', 'ALIAS_UPD', 'CHOOSING', 'ARCHIVING', 'AUTHO_CONF', 'AUTHORIZED') THEN 'OTHERS'
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
            highLevel.refused_rate = ratio
            highLevel.transaction_count_refused = nb_transaction
            highLevel.total_amount_refused = total_amount
    
    return highLevel

def request_data2(date_start, cursor, paytype):
    query = f"""
    SELECT STATE, TRANSACTION_COUNT, TOTAL_AMOUNT, PERCENTAGE FROM HIGH_LEVEL 
    WHERE PAYMENT_TYPE LIKE '{paytype}' 
    AND DATETIME = TO_DATE ('{date_start}', 'YYYY-MM-DD HH24:MI:SS') 
    """

    cursor.execute(query)
    highLevel = HighLevelData2()

    for row in cursor.fetchall():
        status, nb_transaction, total_amount, ratio = row
        match status:
            case 'PAID':
                highLevel.paid_rate = ratio
                highLevel.paid_transaction_count = nb_transaction
                highLevel.paid_total_amount = total_amount
            case 'UNPAID':
                highLevel.unpaid_rate = ratio
                highLevel.unpaid_transaction_count = nb_transaction
                highLevel.unpaid_total_amount = total_amount
            case 'ABANDONED':
                highLevel.abandoned_rate = ratio
                highLevel.abandoned_transaction_count = nb_transaction
                highLevel.abandoned_total_amount = total_amount
    # print(highLevel._convert_to_csv())
    return highLevel

def extract_data(cursor, fname, paytype='OGONE_HID'):
    data = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # make a loop to request data of the every 30 minutes of each day for the month of January
    # for month in range(1, 13):
    month = 9
    for i in range(1, days_in_month[month-1]+1):
        for j in range(0, 23):
            # loop between 00 and 30, then 30 and 00 next hour
            if j % 2 == 0:
                date_start = f"2021-{month}-{i} {j}:00:00"
                date_end = f"2021-{month}-{i} {j}:30:00"
            else:
                date_start = f"2021-{month}-{i} {j}:30:00"
                date_end = f"2021-{month}-{i} {j+1}:00:00"

            data.append(request_data(date_start, date_end, cursor))

    count = 0
    # write the data into a csv file
    with open(f'data/{fname}', 'w') as f:
        # write a header
        f.write("success_rate,transaction_count_success,total_amount_success,transaction_count_refused,total_amount_refused,incident\n")
        
        for highLevel in data:
            # check if the highLevel is empty
            if highLevel.paid_rate == 0.0:
                count+=1
            else:
                f.write(highLevel._convert_to_csv())

    print(f"Number of empty highLevel: {count}/{len(data)}")


def extract_data2(cursor, fname, paytype='OGONE_HID'):
    data = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30]

    for month in range(1, 10):
        print("Month: ", month)
        for i in range(1, days_in_month[month-1]+1):
            for j in range(0, 24):
                date_start = f"2023-{month}-{i} {j}:00:00"
                # print(date_start)

                data.append(request_data2(date_start, cursor, paytype))

    count = 0
    # write the data into a csv file
    with open(f'data/{fname}', 'w') as f:
        # write a header
        f.write("paid_rate,paid_transaction_count,paid_total_amount,unpaid_rate,unpaid_transaction_count,unpaid_total_amount,abandoned_rate,abandoned_transaction_count,abandoned_total_amount\n")
        
        for highLevel in data:
            f.write(highLevel._convert_to_csv())

    print(f"Number of empty highLevel: {count}/{len(data)}")

def insert_data(cursor, filename):
    # create the table HIGH_LEVEL
    # query = """CREATE TABLE HIGH_LEVEL (
    # HOUR DATE,
    # PAYMENT_TYPE VARCHAR2(50),
    # STATE VARCHAR2(50),
    # TRANSACTION_COUNT NUMBER,
    # TOTAL_AMOUNT NUMBER,
    # PERCENTAGE NUMBER)"""

    # cursor.execute(query)

    with open(filename, 'r') as f:
        # read each line that is a insert request
        for line in f.readlines():
            # print(f"'{line[:-2]}'")
            cursor.execute(line[:-2])


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

# insert_data(cursor, "../database/data/01-09-23_30-09-23_stats-pay-ratios.sql")
# insert_data(cursor, "../database/data/01-01-23_31-08-23_stats-pay-ratios.sql")

# extract_data2(cursor, 'data_adyen.csv', 'ADYEN_JSHD')
extract_data2(cursor, 'data_ogone.csv', 'OGONE_HID')

connection.commit()
cursor.close()
connection.close()