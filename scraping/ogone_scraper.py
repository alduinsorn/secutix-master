
# il y a plusieurs type d'incidents
# 1. Scheduled Maintenance (green)
    # dans ce cas la date et l'heure sont donnés dans la step "Scheduled" du début et de fin
    # il peut etre possible que le scheduled dure plus longtemps ?
# 2. Service Disruption (yellow)
    # recup la date et l'heure dans le step "Identified"
    # puis la fin dans "Resolved"
# 3. Restored (black)
    # contient directement les 2 dates ou heures
    # "(09:28-09:40 & 09:47-10:05 CEST)" ou "23:46 19/09/2023 and 02:06 CEST today 20/09/2023"
# 4. Other (orange) (ex: "https://ingenico-ogone.statuspage.io/incidents/wxhrnvvp460k")
    # souvent moins important, car c'est un type spécial(App anti fraud)

import json
import os

from time import sleep
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from time import time as get_time
from datetime import datetime

from utils.utils import clean_special_characters, Incident
from utils.datetime_utils import DatetimeUtils, OUTPUT_DATETIME_FORMAT, MONTHS
from utils.nlp_utils import NLPUtils

class OgoneScraper:
    def __init__(self, driver, filename="ogone_incidents.json", json_export=True):
        self.driver = driver
        self.incidents_dict = {}
        self.json_export = json_export
        self.filename = filename
        
        self.base_url = "https://ingenico-ogone.statuspage.io/history?page="

        # Example : 'Jul 16, 2023 - 11:29 CEST'
        self.INPUT_DATETIME_FORMAT = "%b %d, %Y - %H:%M"

        self.nlp_utils = NLPUtils("ogone")
    
    def _scrap_history(self):
        print("Begin scraping")
        start_time = get_time()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        data_file = os.path.join(data_dir, self.filename)
        
        # Search for the file
        try:
            with open(data_file, 'r') as f:
                self.incidents_dict = json.load(f)
        except Exception as e:
            print("No JSON file found, start the scrapping from the beginning")

        last_year_scrapped = -1
        last_month_index_scrapped = -1
        if len(self.incidents_dict) > 0:
            last_year_scrapped = list(self.incidents_dict.keys())[0]
            if len(self.incidents_dict[last_year_scrapped]) > 0:
                last_month_scrapped = list(self.incidents_dict[last_year_scrapped].keys())[0]
                last_month_index_scrapped = MONTHS.index(last_month_scrapped)

        if last_year_scrapped != -1 and last_month_index_scrapped != -1:
            # remove the last month scrapped in the dict
            self.incidents_dict[last_year_scrapped].pop(last_month_scrapped)
            # convert dictionary key year to int
            self.incidents_dict = {int(k): v for k, v in self.incidents_dict.items()}


        page_id = 1
        condition = True
        count_empty = 0

        while condition: # scrap page by page (3 months per page)
            # if we have 4 empty pages in a row, we stop the scraping because we are at the end of the incidents
            if count_empty >= 4: 
                condition = False
                break
            
            print(f"\n\nScraping page {page_id} -> {self.base_url}{page_id}")
            _, soup = self._parse_url(f"{self.base_url}{page_id}", is_page_incident=False)
            
            months_content = soup.find_all('div', class_='month')            
            
            print(f"\nPage {page_id} scrapped, with {len(months_content)} months")
            
            for month_content in months_content:
                month, year, incidents = self._parse_month_part(month_content)
                
                # if year == 2023 and month == "november":
                #     condition = False
                #     break

                # test if the year and the current month is the one before the last one scrapped we stop the scraping
                # we wait for the next month to be scrapped to be sure that we don't miss any incidents
                month_index = -1
                try:
                    month_index = MONTHS.index(month)
                except Exception as e:
                    pass
                if int(last_year_scrapped) == int(year) and month_index == last_month_index_scrapped - 1:
                    condition = False
                    break
                
                # if their is no incidents, we increment the count_empty -> if we have 4 empty pages in a row, we stop the scraping because we are at the end of the incidents
                if len(incidents) == 0:
                    count_empty += 1

                # for every incident, we need to follow the link and scrap the page we are redirected to
                for incident in incidents:
                    _, incident_soup = self._parse_url(incident['href'], is_page_incident=True)
                    
                    # force to use a lambda function to find the div because this div contains a 4th class that could be different                    
                    incident_title = incident_soup.find('div', class_=lambda classes: all(x in classes.split() for x in ['color-primary', 'incident-name', 'whitespace-pre-wrap'])).text
                    
                    print(f"Incident {incident_title}")
                    
                    # for every incidents, there is one or more <div> tag with a class called "row update-row" that contains the steps of the incident (Resolved, Monitoring, Identified, Investigating). We need to get them all in a list and then go through them
                    incident_steps = incident_soup.find_all('div', class_='row update-row')
                    
                    incident_obj = Incident(title=incident_title, identified_datetime="", resolved_datetime="", services=[], payment_methods=[])
                    raw = ""
                    steps_aborted = 0 # if during the gathering of the data we discover that the incident isn't an incident, we abort the appending of the incident
                    
                    # for every step, we need to extract the right information
                    for incident_step in incident_steps:
                        step_title, step_text, step_time = self._get_n_clean_attributs(incident_step)
                        incident_obj.card_datetime = DatetimeUtils.convert_to_date(step_time, self.INPUT_DATETIME_FORMAT)
                        raw += f" {step_title} ( {step_text})\n"

                        elem_date, times_found = self._search_dates_n_times(step_text, incident_obj)
                        if len(times_found) == 0: # if there is no time found, we need to abort the step
                            steps_aborted += 1
                            print(f"No time found, aborting step {steps_aborted}")
                            continue
                        
                        self._action_depending_type_incident(incident_obj, step_title, elem_date, times_found)

                    incident_obj.raw = raw
                    
                    services_cleaned, payment_methods = self.nlp_utils._search_services([incident_obj.raw])
                    incident_obj.services = services_cleaned
                    incident_obj.payment_methods = payment_methods
                    
                        
                    if steps_aborted != len(incident_steps):
                        self.incidents_dict[year][month].append(incident_obj)
                    
                    # condition = False
                    # break
                
            page_id += 1


        print(f"\n\nScraping done in {get_time() - start_time} seconds")
        
        self._export_incidents(data_file)
        
        return -1
    
    def _parse_url(self, url: str, is_page_incident: bool, wait_time: int = 2):
        content = None; soup = None
        
        try:
            self.driver.get(url)
            
            if not is_page_incident:             
                try:
                    # every page contains 3 months of incidents, so we need to click on the 3 button to expand the
                    button_show_all = self.driver.find_elements(By.XPATH, '//div[contains(@class, "expand-incidents font-small border-color color-secondary custom-focus")]')
                    
                    for btn in button_show_all:
                        btn.click()
                        sleep(wait_time)
            
                except Exception as e:
                    print("No more elements for this month")
            # else:
            #     sleep(wait_time)
                
            
            content = self.driver.page_source
            soup = BeautifulSoup(content, 'lxml')
        
        except Exception as e:
            print(f"Error: {e}")
            exit()
        
        
        return content, soup

    def _get_n_clean_attributs(self, incident_step: BeautifulSoup):
        step_title = incident_step.find('div', class_='update-title span3 font-large').text
        step_text = incident_step.find('span', class_='whitespace-pre-wrap')
        step_text = step_text.get_text(separator="\n")
        step_time = incident_step.find('div', class_='update-timestamp font-small color-secondary').text
        # clean the text
        step_title = clean_special_characters(step_title)
        step_text = clean_special_characters(step_text)
        step_time = DatetimeUtils.search_datetime(step_time)
        step_time = clean_special_characters(step_time) 

        return step_title, step_text, step_time
    
    def _search_dates_n_times(self, step_text: str, incident_obj: Incident):
        # get the dates from the text, but keep only the first one
        dates_found = DatetimeUtils.search_dates([step_text])
        elem_date = dates_found[0] if len(dates_found) > 0 else incident_obj.card_datetime
        # search times
        times_found = DatetimeUtils.search_times([step_text], elem_date, self.INPUT_DATETIME_FORMAT)
        # remove the duplicates and sort the times
        times_found.sort()

        return elem_date, times_found

    def _action_depending_type_incident(self, incident_obj: Incident, step_title: str, elem_date: datetime, times_found: list[str]):
        ### Python 3.10 ###
        # match step_title:
        #     case "Resolved":
        #         # create the datetime object with the date from elem_date and the time from times_found
        #         incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
        #         incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)
        #     case "Identified":
        #         incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
        #         pass
        #     case "Investigating": # only in orange incidents
        #         pass                            
        #     case "Monitoring": # not usefull for us
        #         pass
        #     case "Scheduled":
        #         # create the datetime object with the date from elem_date and the time from times_found
        #         incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
        #         incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)

        #     case _:
        #         pass

        if step_title == "Resolved":
            # create the datetime object with the date from elem_date and the time from times_found
            incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
            incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)
        elif step_title == "Identified":
            incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
        elif step_title == "Investigating":  # only in orange incidents
            pass
        elif step_title == "Monitoring":  # not useful for us
            pass
        elif step_title == "Scheduled":
            # create the datetime object with the date from elem_date and the time from times_found
            incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
            incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)
        else:
            pass

    def _export_incidents(self, data_file: str):
        if self.json_export:
            print(self.incidents_dict.items())
            # sort the dict by year and month
            self.incidents_dict = dict(sorted(self.incidents_dict.items(), key=lambda x: (x[0], x[1])))
            # if a year or a month contains no incidents, we remove it
            # save the year and month in a list to avoid the error "RuntimeError: dictionary changed size during iteration"
            years_to_remove = []
            for year, months in self.incidents_dict.items():
                months_to_remove = []
                for month, incidents in months.items():
                    if len(incidents) == 0:
                        months_to_remove.append(month)
                for month in months_to_remove:
                    months.pop(month)
                if len(months) == 0:
                    years_to_remove.append(year)
            
            print("Starting exporting data into JSON file.")
            with open(data_file, 'w') as output_file:
                json.dump(self.incidents_dict, output_file, indent=4, default=Incident._to_dict)
        else:
            # print nicely the dict
            for year, months in self.incidents_dict.items():
                print(f"\n\nYear {year}")
                for month, incidents in months.items():
                    print(f"\nMonth {month}")
                    for incident in incidents:
                        print(incident)

    def _parse_month_part(self, month_content: BeautifulSoup):
        # the month is contains in the <h4> tag, we need to split the text to get only the month and discard the year
        month_title = month_content.find('h4').text

        # assign to variable month the first part of the split and the rest to year
        print(month_title, month_title.split())
        title_arr = month_title.strip().split()
        month = title_arr[0].lower()
        year = int(title_arr[1])

        # check if 'year' exist in the dict, if not create it
        if year not in self.incidents_dict:
            self.incidents_dict[year] = {}
        # create the current month in the dict
        self.incidents_dict[year][month] = []

        # in every month, get all the <a> tag that are the links to the incidents
        incidents = month_content.find_all('a')
        
        print(f"\nMonth {month}, year {year} with {len(incidents)} incidents")

        return month, year, incidents