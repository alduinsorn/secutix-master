import json
import os
import re

import time
import datetime
from calendar import month_name


from utils import *


class AdyenScrapper:
    def __init__(self, driver, filename="adyen_incidents.json", json_export=True, common_words_found=['CEST', 'CET', 'Adyen Support', 'Status Page', 'Please']):
        self.driver = driver
        self.incidents_dict = {}
        self.json_export = json_export
        self.filename = filename
        self.common_words = common_words_found
        
        self.base_url = "https://status.adyen.com/incident-history"
        
        
    ### The 3 main functions ###
    def _scrap_adyen_history(self):
        start_time = time.time()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        data_file = os.path.join(data_dir, self.filename)
        
        # Search for the file
        try:
            with open(data_file, 'r') as f:
                self.incidents_dict = json.load(f)
        except Exception as e:
            print("No JSON file found, start the scrapping from the beginning")

        today_year = int(datetime.datetime.now().strftime('%Y'))
        today_month_name = datetime.datetime.now().strftime('%B')
        id_today_month = list(month_name).index(today_month_name)    

        
        if len(self.incidents_dict) == 0: # Never scrapped
            continue_scrapping = True
            current_year = today_year
            while continue_scrapping:
                count_empty_months = self._scrap_adyen_months(0, 12, current_year)
                if count_empty_months >= 12: # if we have a complete year without any incidents -> we consider that we scrapped everything
                    continue_scrapping = False
                    
                current_year -= 1
        
        else: # Continue the scrapping
            years = self.incidents_dict.keys()
            last_year_scrapped = max([int(y) for y in years])
            
            months = self.incidents_dict[str(last_year_scrapped)].keys()
            id_last_month = max([MONTHS.index(m) for m in months])

            if today_year == last_year_scrapped: # Possibility 1: scrap only in the same year
                self._scrap_adyen_months(id_last_month, id_today_month, today_year)
            elif abs(today_year - last_year_scrapped) == 1: # Possibility 2: scrap between two years
                self._scrap_adyen_months(0, id_today_month, today_year)
                self._scrap_adyen_months(id_last_month, 12, last_year_scrapped)
            else: # Possibility 3: scrap multiple year considering start and end months
                self._scrap_adyen_months(0, id_today_month, today_year)
                
                # get every years between today and last scrapping
                for i in range(1, abs(today_year - last_year_scrapped)): 
                    year_to_scrap = today_year - i # compute the year to scrap depending on the today year and the number of years between today and the last scrapping 
                    self._scrap_adyen_months(0, 12, year_to_scrap)
                
                self._scrap_adyen_months(id_last_month, 12, last_year_scrapped)
        
        
        execution_time = time.time() - start_time
        # compute the number of element inside the dictionnary
        incidents_count = 0
        for year, months in self.incidents_dict.items():
            for month, incidents in months.items():
                incidents_count += len(incidents)
        
        print(f"""Scrapping finished:
            - Time taken:                {int(execution_time // 60)} min {int(execution_time % 60)} sec. 
            - Total number of incidents: {incidents_count}""")

        if self.json_export:
            print("Starting exporting data into JSON file.")
            with open(data_file, 'w') as output_file:
                json.dump(self.incidents_dict, output_file, indent=4, default=Incident._to_dict)

    def _scrap_adyen_months(self, id_month_start, id_month_end, year):
        count_empty_month = 0
        self.incidents_dict[year] = {}
        for id_month in range(id_month_start, id_month_end):
            
            month = MONTHS[id_month]
            
            url = f"{self.base_url}/{month}-{year}"
            print("Currently scrapping URL:", url)
            _, soup = self._parse_url(url)
            
            # contains the incident cards for the month
            incidents_cards = soup.find_all('div', class_='card')
            print("Number of incidents", len(incidents_cards))
            
            # nothing in this page, skip it -> avoid aving empty months inside the json
            if len(incidents_cards) == 0: 
                count_empty_month += 1
                continue
            
            self.incidents_dict[year][month] = []

            for id_card, card in enumerate(incidents_cards):
                # print(f"\nCard id : {id_card}")
                
                # get the different parts of the incident (Title, Resolved, Identified)
                parts = card.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['status-item', 'ds-width-full'])
                
                # we don't keep this for now -> and avoid error
                if len(parts) < 1:
                    print("No incident or useless one")
                    break

                incident_obj = Incident(services=[], identified_datetime=[], resolved_datetime=[]) # force to reset or cache do some shit

                for id_part, part in enumerate(parts):
                    part_title = part.find("span", {"class": ["ds-margin-left-12", "ds-text", "ds-font-weight-bold", "ds-color-black"]})

                    if not part_title:
                        print("Error, no text inside this part")
                        continue
                    
                    part_title = part_title.text
                    
                    if id_part == 0: # title of the incident, always the first part
                        incident_date = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]}).text

                        incident_obj.title = part_title
                        incident_obj.card_datetime = extract_word(incident_date.split(), -1, -1) # remove the CEST or other time zone
                        incident_obj.incident_type = IncidentType.get_incident_type(part_title)
                        continue
                    
                    self._recover_parts_infos(part, part_title, incident_obj)
                
                self.incidents_dict[year][month].append(incident_obj)
            # return 12
            
            
        return count_empty_month

    def _recover_parts_infos(self, part, part_title, incident_obj):
        '''
        Main function that recover the different informations needed by calling the appropriate functions
        
        Parameters:
            part (BeautifulSoup): The part HTML, can be search.
            part_title (str): The title of the part.
            incident_obj (Incident): The letters sequence to convert.

        Returns:
            bool: The part contains or not information (<p> tag).
        '''
        elem_desc_all = part.find_all("p")
        if len(elem_desc_all) == 0: # sometimes the Resolved part has no text -> not an error, just discard it
            print(f"No <p> in the part: {part_title}") # \n{incident_obj}
            return False
        
        
        # Need to get the part date to find the year or always the today year if we found a date
        part_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]}).text
        part_date = extract_word(part_date.split(), -1, -1) # remove 'CEST'
        part_date = DatetimeUtils.convert_to_date(part_date, INPUT_DATETIME_FORMAT) # create a datetime object
        
        # Check if the date of the incident is in the description or take the part global time
        dates_found = DatetimeUtils.search_dates(elem_desc_all) 
        if len(dates_found) > 0:
            elem_date = dates_found[0] # take the first date found, discards other for now
            elem_date = datetime.datetime(
                year = part_date.year,
                month = elem_date.month,
                day = elem_date.day,
                hour = elem_date.hour,
                minute = elem_date.minute
            )
        else:
            elem_date = part_date

        
        times = DatetimeUtils.search_times(elem_desc_all, elem_date)
        match part_title:
            case PartType.RESOLVED.title:
                incident_obj.resolved_datetime = times
            case PartType.IDENTIFIED.title:
                incident_obj.identified_datetime = times
            case _: pass # do nothing because the Updated aren't relevant in our case
        
        
        part_raw = self._retrieve_raw_desc(part, part_title)
        incident_obj.raw += part_raw
        
        services_found = search_services(elem_desc_all)
        services_cleaned = clean_services_found(services_found, self.common_words)
        incident_obj.services = services_cleaned
        return True

    def _retrieve_raw_desc(self, part, part_title):
        all_elem_desc = part.find_all("p")
        raw = f"{part_title}("
        for elem in all_elem_desc:
            raw += elem.text
        raw += ")\n"
        return raw


    def _parse_url(self, url, wait_time=3):
        '''
        Go on the url given to get all the html data contains on it, in addition it clicks on a button to get all the incidents of the month.
        
        Parameters:
            url (str): The url to scrap.
            wait_time (int): The wait time before scrapping and after the click on the button.
            
        Returns:
            tuple(str, BeautifulSoup): The data from the page in raw and object form.
        '''
        self.driver.implicitly_wait(wait_time)
        content = None; soup = None

        try:
            self.driver.get(url)

            try:
                # besoin de cliquer sur un bouton pour afficher tout les incidents
                button_show_all = self.driver.find_element(By.XPATH, '//button[contains(@class, "ds-button-link ds-button-link--primary ds-button-link--green")]')   

                if button_show_all:
                    button_show_all.click()
                    sleep(wait_time) # sleep require or page isn't complete
            except Exception as e:
                print("No more elements for this month")
                

            content = self.driver.page_source
            soup = BeautifulSoup(content, 'lxml')

        except Exception as e:
            print(f"Error: {e}")
            exit()

        return content, soup