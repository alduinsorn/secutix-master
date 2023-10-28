
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


from utils.utils import clean_special_characters, Incident
from utils.datetime_utils import DatetimeUtils, OUTPUT_DATETIME_FORMAT
from utils.nlp_utils import search_services

class OgoneScraper:
    def __init__(self, driver, filename="ogone_incidents.json", json_export=True, common_words_found=['CEST', 'CET', 'Status Page', 'Please']):
        self.driver = driver
        self.incidents_dict = {}
        self.json_export = json_export
        self.filename = filename
        self.common_words = common_words_found
        
        self.base_url = "https://ingenico-ogone.statuspage.io/history?page=" # add the number of the page 

        # Jul 16, 2023 - 11:29 CEST
        self.INPUT_DATETIME_FORMAT = "%b %d, %Y - %H:%M"
    
    def _scrap_history(self):
        print("Begin scraping")
        start_time = get_time()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        data_file = os.path.join(data_dir, self.filename)
        
        page_id = 1
        condition = True
        count_empty = 0

        while condition: # scrap page by page (3 months per page)
            if count_empty >= 4:
                condition = False
                break
            
            print(f"\n\nScraping page {page_id} -> {self.base_url}{page_id}")
            _, soup = self._parse_url(f"{self.base_url}{page_id}", page_incident=False)
            
            months_content = soup.find_all('div', class_='month')            
            
            print(f"\nPage {page_id} scrapped, with {len(months_content)} months")
            
            for month_content in months_content:
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
                
                if year == 2022:
                    condition = False
                    break

                if len(incidents) == 0:
                    count_empty += 1
                

                # for every incident, we need to follow the link and scrap the page we are redirected to
                for idx, incident in enumerate(incidents):
                    _, incident_soup = self._parse_url(incident['href'], page_incident=True)
                    
                    # force to use a lambda function to find the div because this div contains a 4th class that could be different                    
                    incident_title = incident_soup.find('div', class_=lambda classes: all(x in classes.split() for x in ['color-primary', 'incident-name', 'whitespace-pre-wrap'])).text
                    
                    print(f"Incident {incident_title}")
                    
                    # for every incidents, there is one or more <div> tag with a class called "row update-row" that contains the steps of the incident (Resolved, Monitoring, Identified, Investigating). We need to get them all in a list and then go through them
                    incident_steps = incident_soup.find_all('div', class_='row update-row')
                    
                    incident_obj = Incident(title=incident_title, identified_datetime="", resolved_datetime="", services=[], payment_methods=[])
                    raw = ""
                    # if during the gathering of the data we discover that the incident isn't an incident, we abort the appending of the incident
                    steps_aborted = 0
                    
                    # for every step, we need to extract the right information
                    for incident_step in incident_steps:
                        step_title = incident_step.find('div', class_='update-title span3 font-large').text
                        step_text = incident_step.find('span', class_='whitespace-pre-wrap').text
                        step_time = incident_step.find('div', class_='update-timestamp font-small color-secondary').text
                        
                        # clean the text
                        step_title = clean_special_characters(step_title)
                        step_text = clean_special_characters(step_text)
                        step_time = DatetimeUtils.search_datetime(step_time)
                        step_time = clean_special_characters(step_time) 

                        incident_obj.card_datetime = DatetimeUtils.convert_to_date(step_time, self.INPUT_DATETIME_FORMAT)

                        raw += f"{step_title}({step_text})\n"


                        # get the dates from the text, but keep only the first one
                        dates_found = DatetimeUtils.search_dates([step_text])
                        elem_date = dates_found[0] if len(dates_found) > 0 else incident_obj.card_datetime
                        # print("Step_text:", step_text)
                        # search times
                        times_found = DatetimeUtils.search_times([step_text], elem_date, self.INPUT_DATETIME_FORMAT)
                        # remove the duplicates and sort the times
                        times_found.sort()
                        if len(times_found) == 0: # if there is no time found, we need to abort the step
                            steps_aborted += 1
                            print(f"No time found, aborting step {steps_aborted}")
                            continue
                        # else:
                        #     print(f"Time found: {times_found}")



                        match step_title:
                            case "Resolved":
                                # create the datetime object with the date from elem_date and the time from times_found
                                incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
                                incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)
                            case "Identified":
                                incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
                                pass
                            case "Investigating": # only in orange incidents
                                pass                            
                            case "Monitoring": # not usefull for us
                                pass
                            case "Scheduled":
                                # create the datetime object with the date from elem_date and the time from times_found
                                incident_obj.identified_datetime = DatetimeUtils.create_datetime(elem_date, times_found[0], OUTPUT_DATETIME_FORMAT)
                                incident_obj.resolved_datetime = DatetimeUtils.create_datetime(elem_date, times_found[-1], OUTPUT_DATETIME_FORMAT)

                            case _:
                                pass

                    incident_obj.raw = raw
                    
                    services_cleaned = search_services([incident_obj.raw], self.common_words)
                    incident_obj.services = services_cleaned
                    
                        
                    if steps_aborted != len(incident_steps):
                        self.incidents_dict[year][month].append(incident_obj)
                    
                    # condition = False
                    # break
                
            page_id += 1


        print(f"\n\nScraping done in {get_time() - start_time} seconds")
        
        # print every incidents
        # for year, months in self.incidents_dict.items():
        #     for month, incidents in months.items():
        #         print(f"\n\n{month} {year}")
        #         for incident in incidents:
        #             print(incident)        

        # go through every incidents and verify that the variable is of type Incident
        # for year, months in self.incidents_dict.items():
        #     for month, incidents in months.items():
        #         for incident in incidents:
        #             if not isinstance(incident, Incident):
        #                 print("Error, the incident is not of type Incident")
        #                 print(incident)
        #                 exit()


        if self.json_export:
            # sort the dict by year and month
            self.incidents_dict = dict(sorted(self.incidents_dict.items(), key=lambda x: (x[0], x[1])))
            

            print("Starting exporting data into JSON file.")
            with open(data_file, 'w') as output_file:
                json.dump(self.incidents_dict, output_file, indent=4, default=Incident._to_dict)

        
        return -1
    
    def _parse_url(self, url, page_incident, wait_time=2):
        content = None; soup = None
        
        try:
            self.driver.get(url)
            
            if not page_incident:             
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
