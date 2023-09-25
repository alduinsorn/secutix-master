import json
import os

from time import sleep
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from time import time as get_time


from utils.utils import setup_driver


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


            # get the div month that have a single class called "month"
            
            months_content = soup.find_all('div', class_='month')
            
            if len(months_content) == 0:
                condition = False
                break
            
            print(f"Page {page_id} scrapped, with {len(months_content)} months")
            
            for month_content in months_content:
                # in every month, get all the <a> tag that are the links to the incidents
                incidents = month_content.find_all('a')
                
                print(f"Month {month_content.find('h4').text} with {len(incidents)} incidents")
                
                # for every incident, we need to follow the link and scrap the page we are redirected to

                _, incident_soup = self._parse_url(incidents[0]['href'], page_incident=True)
                
                # incident_title = incident_soup.find('div', class_='color-primary incident-name whitespace-pre-wrap').text
                
                incident_title = incident_soup.find('div:is(.color-primary, .incident-name, .whitespace-pre-wrap)')

                
                print(f"Incident {incident_title}")
                
                # for every incidents, there is one or more <div> tag with a class called "row update-row" that contains the steps of the incident (Resolved, Monitoring, Identified, Investigating). We need to get them all in a list and then go through them
                incident_steps = incident_soup.find_all('div', class_='row update-row')
                
                # for every step, we need to get the date and the description
                for incident_step in incident_steps:
                    step_title = incident_step.find('div', class_='update-title span3 font-large').text
                    step_text = incident_step.find('span', class_='whitespace-pre-wrap').text
                    step_time = incident_step.find('div', class_='update-timestamp font-small color-secondary').text
                    
                    print(f"Step {step_title} at {step_time} with text: {step_text}")
                
                # il y a plusieurs type d'incidents
                # 1. Scheduled Maintenance (green)
                # 2. Service Disruption (yellow)
                # 3. Restored (black)
                # 4. Other (orange) (ex: "https://ingenico-ogone.statuspage.io/incidents/wxhrnvvp460k")
                
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
