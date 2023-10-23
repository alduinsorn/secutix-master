import unittest
from datetime import datetime
from utils.datetime_utils import DatetimeUtils

class TestDatetimeUtils(unittest.TestCase):

    def test_search_times(self):
        elem_desc_all = ["12:15 CEST"]
        elem_date = datetime(2023, 10, 18, 10, 0)  # Replace with your desired date
        input_datetime_format = '%Y-%m-%d %H:%M'

        result = DatetimeUtils.search_times(elem_desc_all, elem_date, input_datetime_format)
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0], str))

    def test_clean_time(self):
        time_str = "12:15"
        general_parsed_datetime = datetime(2023, 10, 18, 10, 0)  # Replace with your desired date

        result = DatetimeUtils.clean_time(time_str, general_parsed_datetime)
        self.assertTrue(isinstance(result, str))  # Assuming your clean_time function returns a string

    def test_search_dates(self):
        elem_desc_all = ["October 18th, 2023"]
        result = DatetimeUtils.search_dates(elem_desc_all)
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0], datetime))

    def test_convert_to_date(self):
        text = "2023-10-18 15:30"
        date_format = '%Y-%m-%d %H:%M'

        result = DatetimeUtils.convert_to_date(text, date_format)
        self.assertTrue(isinstance(result, datetime))

    def test_convert_to_str(self):
        date = datetime(2023, 10, 18, 15, 30)
        date_format = '%Y-%m-%d %H:%M'

        result = DatetimeUtils.convert_to_str(date, date_format)
        self.assertTrue(isinstance(result, str))

    def test_get_today_date(self):
        result = DatetimeUtils.get_today_date()
        self.assertTrue(isinstance(result, datetime))

    def test_get_month_id(self):
        date = datetime(2023, 10, 18)
        result = DatetimeUtils.get_month_id(date)
        self.assertTrue(isinstance(result, int))

if __name__ == '__main__':
    unittest.main()
