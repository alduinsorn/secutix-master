from re import findall as re_findall, IGNORECASE
from datetime import datetime
from dateutil.parser import parse
from calendar import month_name


# Regex formula detecting hours in different format and typing error
TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])\s*[:.]\s*[0-5][0-9](?:[A-Za-z]{3-4})?'
# Regex formula detecting writted date in 2 different common formats 
DATE_REGEX_1 = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*\b'
DATE_REGEX_2 = r'\b\d{1,2}(?:st|nd|rd|th)?\s*(?:of)?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'

# Datetime format for the datetime library, for now based on the Adyen data
INPUT_DATETIME_FORMAT = '%B %d %Y, %H:%M'
INPUT_TIME_FORMAT = '%H:%M'
OUTPUT_DATETIME_FORMAT = '%Y-%m-%d %H:%M'

# name of the months in lowercase (month_name from calendar have first letter in uppercase)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

class DatetimeUtils:
    # Function that search any times given in the description like "12:15 CEST" and then assign it to the correct datetime attribut
    def search_times(elem_desc_all, elem_date):
        # General datetime from the part
        general_parsed_datetime = elem_date if isinstance(elem_date, datetime) else datetime.strptime(elem_date, INPUT_DATETIME_FORMAT)
        datetime_arr = []
        
        # For every <p> tag in the part description
        for desc in elem_desc_all:
            times_arr = re_findall(TIME_REGEX, desc.text) # get all the time
            for time_str in times_arr:
                time_found = DatetimeUtils.clean_time(time_str, general_parsed_datetime)
                if time_found:
                    datetime_arr.append(time_found)

        datetime_arr = list(set(datetime_arr))
        return [datetime.strftime(general_parsed_datetime, INPUT_DATETIME_FORMAT)] if len(datetime_arr) == 0 else datetime_arr

    def clean_time(time_str, general_parsed_datetime):
        if not time_str: return None
        time_str = time_str.replace('.', ':').replace(" ", "") # sometimes they have miss typed

        time_var = datetime.strptime(time_str, INPUT_TIME_FORMAT)
        
        new_datetime = datetime(
            year = general_parsed_datetime.year,
            month = general_parsed_datetime.month,
            day = general_parsed_datetime.day,
            hour = time_var.hour,
            minute = time_var.minute
        )
        return new_datetime.strftime(OUTPUT_DATETIME_FORMAT)

    def search_dates(elem_desc_all):
        parsed_dates = []
        for elem_desc in elem_desc_all:
            # search for the date in text -> can vary a lot
            date_matches = re_findall(DATE_REGEX_1, elem_desc.text, IGNORECASE)
            date_matches.extend(re_findall(DATE_REGEX_2, elem_desc.text, IGNORECASE))
            for date_match in date_matches:
                parsed_date = parse(date_match, fuzzy=True)
                parsed_dates.append(parsed_date)
        
        return parsed_dates

    def convert_to_date(text, date_format):
        return datetime.strptime(text, date_format)

    def get_today_date():
        return datetime.now()
    
    def get_month_id(date):
        return list(month_name).index(date.strftime('%B'))
    
