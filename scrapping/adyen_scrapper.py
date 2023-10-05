import json
import os

from time import sleep, time as get_time
from datetime import datetime

from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from bs4.element import ResultSet, Tag

from utils.utils import IncidentType, Incident, PartType, extract_word 
from utils.nlp_utils import search_services
from utils.datetime_utils import DatetimeUtils, MONTHS

from typing import List, Tuple

class AdyenScrapper:
    '''
    
    '''
    def __init__(self, driver, filename: str = "adyen_incidents.json", json_export: bool = True, common_words_found: List[str] = ['CEST', 'CET', 'Adyen Support', 'Status Page', 'Please']):
        self.driver = driver
        self.incidents_dict = {}
        self.json_export = json_export
        self.filename = filename
        self.common_words = common_words_found
        
        self.base_url: str = "https://status.adyen.com/incident-history"
        
        self.INPUT_DATETIME_FORMAT = '%B %d %Y, %H:%M'
        
        
    ### The 3 main functions ###
    def _scrap_adyen_history(self) -> None:
        start_time: float = get_time()
        
        script_dir: str = os.path.dirname(os.path.abspath(__file__))
        data_dir: str = os.path.join(script_dir, "data")
        data_file: str = os.path.join(data_dir, self.filename)
        
        # Search for the file
        try:
            with open(data_file, 'r') as f:
                self.incidents_dict = json.load(f)
        except Exception as e:
            print("No JSON file found, start the scrapping from the beginning")


        today: datetime = DatetimeUtils.get_today_date()
        id_today_month: int = DatetimeUtils.get_month_id(today)
        
        if len(self.incidents_dict) == 0: # Never scrapped
            continue_scrapping: bool = True
            current_year: int = today.year
            while continue_scrapping:
                count_empty_months: int = self._scrap_adyen_months(0, 12, current_year)
                if count_empty_months >= 12: # if we have a complete year without any incidents -> we consider that we scrapped everything
                    continue_scrapping = False
                    
                current_year -= 1
        
        else: # Continue the scrapping
            years: int = self.incidents_dict.keys()
            last_year_scrapped: int = max([int(y) for y in years])
            
            months: List[str] = self.incidents_dict[last_year_scrapped].keys()
            id_last_month: int = max([MONTHS.index(m) for m in months])

            if today.year == last_year_scrapped: # Possibility 1: scrap only in the same year
                self._scrap_adyen_months(id_last_month, id_today_month, today.year)
            elif abs(today.year - last_year_scrapped) == 1: # Possibility 2: scrap between two years
                self._scrap_adyen_months(0, id_today_month, today.year)
                self._scrap_adyen_months(id_last_month, 12, last_year_scrapped)
            else: # Possibility 3: scrap multiple year considering start and end months
                self._scrap_adyen_months(0, id_today_month, today.year)
                
                # get every years between today and last scrapping
                for i in range(1, abs(today.year - last_year_scrapped)): 
                    year_to_scrap: int = today.year - i # compute the year to scrap depending on the today year and the number of years between today and the last scrapping 
                    self._scrap_adyen_months(0, 12, year_to_scrap)
                
                self._scrap_adyen_months(id_last_month, 12, last_year_scrapped)
        
        
        execution_time: float = get_time() - start_time
        # compute the number of element inside the dictionnary
        incidents_count: int = 0
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

    def _scrap_adyen_months(self, id_month_start: int, id_month_end: int, year: int) -> int:
        '''
        Scrap the incidents that happened between the two months given in parameter for the year given.
        
        Parameters:
            id_month_start (int): The id of the month to start the scrapping. ∈ [0, 12[
            id_month_end (int): The id of the month to end the scrapping. ∈ [0, 12[
            year (int): The year to scrap.
            
        Return:
            int: The number of empty months found during the scrapping.        
        '''
        count_empty_month: int = 0
        self.incidents_dict[year]: dict = {}
        for id_month in range(id_month_start, id_month_end):
            
            month: str = MONTHS[id_month]
            
            url: str = f"{self.base_url}/{month}-{year}"
            print("Currently scrapping URL:", url)
            _, soup = self._parse_url(url)
            
            # contains the incident cards for the month
            incidents_cards: ResultSet = soup.find_all('div', class_='card')
            print("Number of incidents", len(incidents_cards))
            
            # nothing in this page, skip it -> avoid aving empty months inside the json
            if len(incidents_cards) == 0: 
                count_empty_month += 1
                continue
            
            self.incidents_dict[year][month] = []

            for id_card, card in enumerate(incidents_cards):
                # print(f"\nCard id : {id_card}")
                
                # get the different parts of the incident (Title, Resolved, Identified)
                parts: List[ResultSet] = card.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['status-item', 'ds-width-full'])
                
                # we don't keep this for now -> and avoid error
                if len(parts) < 1:
                    print("No incident or not taken into account")
                    break

                incident_obj: Incident = Incident(services=[], identified_datetime=[], resolved_datetime=[]) # force to reset or app cache is doing some shit stuff

                for id_part, part in enumerate(parts):
                    part_title: Tag  = part.find("span", {"class": ["ds-margin-left-12", "ds-text", "ds-font-weight-bold", "ds-color-black"]})

                    if not part_title:
                        print("Error, no text inside this part")
                        continue
                    
                    part_title: str = part_title.text
                    
                    if id_part == 0: # title of the incident, always the first part
                        incident_date: str = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]})
                        
                        if not incident_date: 
                            # should be impossible to happen because the time is set when creating the part (automatic by Adyen system), set specific date for this
                            incident_obj.card_datetime = DatetimeUtils.convert_to_str(datetime(1970, 1, 1))
                        else:                        
                            incident_obj.card_datetime = extract_word(incident_date.text.split(), -1, -1) # remove the CEST or other time zone

                        incident_obj.title = part_title
                        incident_obj.incident_type = IncidentType.get_incident_type(part_title)
                        continue
                    
                    
                    if not self._recover_parts_infos(part, part_title, incident_obj):
                        # empty element (no text) but not an error
                        pass
                
                self.incidents_dict[year][month].append(incident_obj)
            # return 12

        return count_empty_month

    def _recover_parts_infos(self, part: ResultSet[str], part_title: str, incident_obj: Incident) -> bool:
        '''
        Main function that recover the different informations needed by calling the appropriate functions
        
        Parameters:
            part (BeautifulSoup): The part HTML, can be search.
            part_title (str): The title of the part.
            incident_obj (Incident): The letters sequence to convert.

        Returns:
            bool: The part contains or not information (<p> tag).
        '''
        elem_desc_all: ResultSet[str] = part.find_all("p")
        if len(elem_desc_all) == 0: # sometimes the Resolved part has no text -> not an error, just discard it
            print(f"No <p> in the part: {part_title}") # \n{incident_obj}
            return False
        
        # Need to get the part date to find the year or always the today year if we found a date
        part_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]})
        
        if not part_date:
            # should be impossible to happen because the time is set when creating the part (automatic by Adyen system), set specific date for this
            part_date = DatetimeUtils.convert_to_str(datetime(1970, 1, 1))
        else:
            part_date = extract_word(part_date.text.split(), -1, -1) # remove 'CEST'
        
        part_date = DatetimeUtils.convert_to_date(part_date, self.INPUT_DATETIME_FORMAT) # create a datetime object
        
        # Check if the date of the incident is in the description or take the part global time
        dates_found = DatetimeUtils.search_dates(elem_desc_all) 
        # if a date is found, create a new datetime object with the year of the part (because the date found doesn't have a year and it will take the current year)
        if len(dates_found) > 0:
            elem_date = dates_found[0] # take the first date found, discards other because impossible to process
            elem_date = datetime(
                year = part_date.year,
                month = elem_date.month,
                day = elem_date.day,
                hour = elem_date.hour,
                minute = elem_date.minute
            )
        else:
            elem_date = part_date

        elem_desc_all_str: List[str] = [elem.text for elem in elem_desc_all]
        times: List[datetime] = DatetimeUtils.search_times(elem_desc_all_str, elem_date, self.INPUT_DATETIME_FORMAT)
        match part_title:
            case PartType.RESOLVED.title:
                incident_obj.resolved_datetime = times
            case PartType.IDENTIFIED.title:
                incident_obj.identified_datetime = times
            case _: pass # do nothing because the Updated cases aren't relevant in our case
        
        
        part_raw: str = self._retrieve_raw_desc(part, part_title)
        incident_obj.raw += part_raw
        
        services_cleaned: List[str] = search_services(elem_desc_all_str, self.common_words)
        incident_obj.services = services_cleaned
        return True

    def _retrieve_raw_desc(self, part: ResultSet[str], part_title: str) -> str:
        '''
        Retrieve the text inside the <p> tag of the part given in parameter
        
        Parameters:
            part (ResultSet[str]): The part HTML, can be search.
            part_title (str): The title of the part.
            
        Returns:
            str: The text inside the <p> tag of the part.
        '''
        all_elem_desc: ResultSet[str] = part.find_all("p")
        if len(all_elem_desc) == 0: 
            return ""
        raw: str = f"{part_title}("
        for elem in all_elem_desc:
            raw += elem.text
        raw += ")\n"
        return raw

    def _parse_url(self, url: str, wait_time: int = 3) -> Tuple[str, BeautifulSoup]:
        '''
        Go on the url given to get all the html data contains on it, in addition it clicks on a button to get all the incidents of the month.
        
        Parameters:
            url (str): The url to scrap.
            wait_time (int): The wait time before scrapping and after the click on the button.
            
        Returns:
            tuple(str, BeautifulSoup): The data from the page in raw and object form.
        '''
        self.driver.implicitly_wait(wait_time)
        content: str = None; soup: BeautifulSoup = None

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