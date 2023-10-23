from re import findall as re_findall, IGNORECASE
from datetime import datetime
from dateutil.parser import parse
from calendar import month_name

from typing import List, Union

# Regex formula detecting hours in different format and typing error
TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])\s*[:.]\s*[0-5][0-9](?:[A-Za-z]{3-4})?'
# Regex formula detecting writted date in 2 different common formats 
DATE_REGEX_1 = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*\b'
DATE_REGEX_2 = r'\b\d{1,2}(?:st|nd|rd|th)?\s*(?:of)?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
# this search for date in the format "dd/mm/yyyy" or "dd-mm-yyyy" or "dd.mm.yyyy"
DATE_REGEX_3 = r'\b\d{1,2}[-./]\d{1,2}[-./]\d{4}\b'

# Datetime format for the datetime library, for now based on the Adyen data
INPUT_TIME_FORMAT = '%H:%M'
OUTPUT_DATETIME_FORMAT = '%Y-%m-%d %H:%M'

# name of the months in lowercase (month_name from calendar have first letter in uppercase)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

class DatetimeUtils:
    '''
    This class is used to parse and clean datetimes from a text or a datetime object
    '''
    
    # Function that search any times given in the description like "12:15 CEST" and then assign it to the correct datetime attribut
    def search_times(elem_desc_all: List[str], elem_date: datetime, input_datetime_format: str) -> List[datetime]:
        '''
        This function search for any time in the description using the TIME_REGEX and then assign it to the correct datetime attribut
        
        Parameters:
            elem_desc_all (List[str]): list of text (description of the incident)
            elem_date (datetime): datetime of the part
            input_datetime_format (str): format of the datetime

        Returns:
            List[datetime]: list of all the datetime found in the description
        '''
        # General datetime from the part
        general_parsed_datetime: datetime = elem_date
        datetime_arr: List[datetime] = []
        
        # For every <p> tag in the part description
        for desc in elem_desc_all:
            times_arr: List[str] = re_findall(TIME_REGEX, desc) # get all the time
            for time_str in times_arr:
                time_found: str = DatetimeUtils.clean_time(time_str, general_parsed_datetime)
                if time_found:
                    datetime_arr.append(time_found)

        datetime_arr: list = list(set(datetime_arr))
        return [datetime.strftime(general_parsed_datetime, input_datetime_format)] if len(datetime_arr) == 0 else datetime_arr

    def clean_time(time_str: str, general_parsed_datetime: datetime) -> Union[datetime, None]:
        '''
        This function clean the time string and convert it to a datetime object
        
        Parameters:
            time_str (str): string containing the time
            general_parsed_datetime (datetime): datetime of the part
            
        Returns:
            datetime: datetime object of the time found
        '''
        
        if not time_str: return None
        time_str: str = time_str.replace('.', ':').replace(" ", "") # sometimes they have miss typed

        time_var: datetime = datetime.strptime(time_str, INPUT_TIME_FORMAT)
        
        new_datetime: datetime = datetime(
            year = general_parsed_datetime.year,
            month = general_parsed_datetime.month,
            day = general_parsed_datetime.day,
            hour = time_var.hour,
            minute = time_var.minute
        )
        return new_datetime.strftime(OUTPUT_DATETIME_FORMAT)

    def search_dates(elem_desc_all: List[str]) -> List[datetime]:
        '''
        This function search for any date in the description using the DATE_REGEX and then store it in a list
        
        Parameters:
            elem_desc_all (List<str>): list of text (description of the incident)
            
        Returns:
            List<datetime>: list of all the date found in the description
        '''
        parsed_dates: List[datetime] = []
        for elem_desc in elem_desc_all:
            # search for the date in text -> can vary a lot
            date_matches: List[str] = re_findall(DATE_REGEX_1, elem_desc, IGNORECASE)
            date_matches.extend(re_findall(DATE_REGEX_2, elem_desc, IGNORECASE))
            date_matches.extend(re_findall(DATE_REGEX_3, elem_desc, IGNORECASE))
            for date_match in date_matches:
                parsed_date: datetime = parse(date_match, fuzzy=True)
                parsed_dates.append(parsed_date)
        
        return parsed_dates

    def convert_to_date(text: str, date_format: str) -> datetime:
        '''
        This function convert a text to a datetime object using the date_format
        
        Parameters:
            text (str): text to convert
            date_format (str): format of the date
            
        Returns:
            datetime: datetime object of the text
        '''
        return datetime.strptime(text, date_format)

    def convert_to_str(date: datetime, date_format: str = OUTPUT_DATETIME_FORMAT) -> str:
        '''
        This function convert a datetime object to a string using the date_format
        
        Parameters:
            date (datetime): datetime object to convert
            date_format (str): format of the date
            
        Returns:
            str: string of the datetime object
        '''
        return datetime.strftime(date, date_format)

    def get_today_date() -> datetime:
        '''
        This function return the current datetime
        
        Returns:
            datetime: current datetime
        '''
        return datetime.now()
    
    def get_month_id(date: datetime) -> int:
        '''
        This function return the id of the month of the given date 
        
        Parameters:
            date (datetime): datetime object
            
        Returns:
            int: id of the month of the given date
        '''
        return list(month_name).index(date.strftime('%B'))
    
