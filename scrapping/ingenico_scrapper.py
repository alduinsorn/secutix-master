import json
import os

from time import sleep
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from time import time as get_time


from utils.utils import clean_special_characters, Incident


class IngenicoScrapper:
    def __init__(self, driver, filename="ingenico_incidents.json", json_export=True, common_words_found=['CEST', 'CET', 'Status Page', 'Please']):
        self.driver = driver
        self.incidents_dict = {}
        self.json_export = json_export
        self.filename = filename
        self.common_words = common_words_found
        
        self.base_url = "https://ingenico-ogone.statuspage.io/history?page=" # add the number of the page 
    
    def _scrap_history(self):
        start_time = get_time()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        data_file = os.path.join(data_dir, self.filename)
        
        page_id = 1
        condition = True
        while condition: # scrap page by page (3 months per page)
            _, soup = self._parse_url(f"{self.base_url}{page_id}", page_incident=False)
            
            # get the year of the 3 months displayed, contains in a <var> tag with the attribut data-var="start-date-year"
            year: int = int(soup.find('var', attrs={'data-var': 'start-date-year'}).text)
            self.incidents_dict[year] = {}

            months_content = soup.find_all('div', class_='month')            
            if len(months_content) == 0:
                condition = False
                break
            
            print(f"\nPage {page_id} scrapped, with {len(months_content)} months")
            
            for month_content in months_content:
                # in every month, get all the <a> tag that are the links to the incidents
                incidents = month_content.find_all('a')
                
                # the month is contains in the <h4> tag, we need to split the text to get only the month and discard the year
                month = month_content.find('h4').text.split()[0].lower()
                self.incidents_dict[year][month] = []
                
                print(f"\nMonth {month}, year {year} with {len(incidents)} incidents")
                
                # for every incident, we need to follow the link and scrap the page we are redirected to
                for incident in incidents:
                    _, incident_soup = self._parse_url(incident['href'], page_incident=True)
                    
                    # force to use a lambda function to find the div because this div contains a 4th class that could be different                    
                    incident_title = incident_soup.find('div', class_=lambda classes: all(x in classes.split() for x in ['color-primary', 'incident-name', 'whitespace-pre-wrap'])).text
                    
                    print(f"Incident {incident_title}")
                    
                    # for every incidents, there is one or more <div> tag with a class called "row update-row" that contains the steps of the incident (Resolved, Monitoring, Identified, Investigating). We need to get them all in a list and then go through them
                    incident_steps = incident_soup.find_all('div', class_='row update-row')
                    
                    incident_obj = Incident(title=incident_title)
                    
                    # for every step, we need to extract the right information
                    for incident_step in incident_steps:
                        step_title = incident_step.find('div', class_='update-title span3 font-large').text
                        step_text = incident_step.find('span', class_='whitespace-pre-wrap').text
                        step_time = incident_step.find('div', class_='update-timestamp font-small color-secondary').text
                        
                        # clean the text
                        step_title = clean_special_characters(step_title)
                        step_text = clean_special_characters(step_text)
                        step_time = clean_special_characters(step_time)
                        
                        print(f"Step {step_title} at {step_time} with text: {step_text}")
                        
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
                    
                    self.incidents_dict[year][month].append(incident_obj)
                    
                    condition = False
                
                
                
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
