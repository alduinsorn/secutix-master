# # General import
# import json
# import os
# import re

# import time
# import datetime
# from calendar import month_name
# from dateutil.parser import parse


# # Scrapper module
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from bs4 import BeautifulSoup

# # Personal files
# from utils import *


# ##### General scrapper functions #####

# def setup_driver(headless=True):
#     options = webdriver.FirefoxOptions()
#     if headless:
#         options.add_argument('--headless')    
    
#     driver = webdriver.Firefox(options=options)

#     return driver

# def parse_url(driver, url, wait_time=2):
#     driver.implicitly_wait(wait_time)
#     content = None; soup = None

#     try:
#         driver.get(url)

#         try:
#             # besoin de cliquer sur un bouton pour afficher tout les incidents
#             button_show_all = driver.find_element(By.XPATH, '//button[contains(@class, "ds-button-link ds-button-link--primary ds-button-link--green")]')   

#             if button_show_all:
#                 button_show_all.click()
#                 time.sleep(wait_time) # sleep require or page isn't complete
#         except Exception as e:
#             print("No element for this month")
            

#         content = driver.page_source
#         soup = BeautifulSoup(content, 'lxml')

#     except Exception as e:
#         print(f"Error: {e}")
#         exit()

#     return content, soup



# ##### Data retrieving for Adyen #####

# def clean_services_found(incident_obj, words_often_found):
#     words_often_found = ['CEST', 'CET', 'Adyen Support']
    
#     cleaned_services = []
#     seen_services = set()
    
#     for service in incident_obj.services:
#         lowercase_service = service.lower()
#         if lowercase_service not in seen_services and service not in words_often_found:
#             cleaned_services.append(service)
#             seen_services.add(lowercase_service)
    
#     incident_obj.services = cleaned_services

# def search_service_n_bank(elem_desc_all, incident_obj):
#     for elem_desc in elem_desc_all:
#         # Add all proper names in the bank arrray -> should do everything -> service and bank
#         proper_names = extract_proper_names(elem_desc.text)
#         for name in proper_names:
#             incident_obj.services.append(name.strip())
    
#     clean_services_found(incident_obj)    


# ### Date and time function associated




# def recover_part_infos(part, part_title, incident_obj):
#     '''
#     Main function that recover the different informations needed by calling the appropriate functions
    
#     Parameters:
#         part (BeautifulSoup): The part HTML, can be search.
#         part_title (str): The title of the part.
#         incident_obj (Incident): The letters sequence to convert.

#     Returns:
#         bool: The part contains or not information (<p> tag).
#     '''
#     elem_desc_all = part.find_all("p")
#     if len(elem_desc_all) == 0: # sometimes the Resolved part has no text -> not an error, just discard it
#         print(f"No <p> in the part: {part_title}\n")
#         return False
    
#     # Check if the date of the incident is in the description or take the part global time
#     elem_date = search_date_in_desc(elem_desc_all)
#     # Need to get the part date to find the year or always the today year
#     part_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]}).text
#     part_date = extract_word(part_date.split(), -1, -1) # remove 'CEST'
#     part_date = datetime.datetime.strptime(part_date, INPUT_DATETIME_FORMAT) # create a datetime object
    
#     if not elem_date:
#         elem_date = part_date
#     else:
#         elem_date = datetime.datetime(
#             year = part_date.year,
#             month = elem_date.month,
#             day = elem_date.day,
#             hour = elem_date.hour,
#             minute = elem_date.minute
#         )
    
#     search_assign_datetime(elem_desc_all, elem_date, incident_obj, part_title)
#     retrieve_raw_desc(part, incident_obj, part_title)
#     search_service_n_bank(elem_desc_all, incident_obj)
    
#     return True

# ### The two main function of the Adyen scrapping

# def scrap_adyen_months(id_month_start, id_month_end, year, incidents_dict, driver):
#     count_empty_month = 0
#     incidents_dict[year] = {}
#     for id_month in range(id_month_start, id_month_end):
        
#         month = MONTHS[id_month]
        
#         url = f"{ADYEN_BASE_URL}/{month}-{year}"
#         print("Currently scrapping URL:", url)
#         _, soup = parse_url(driver, url)
        
#         # contains the incident cards for the month
#         incidents_cards = soup.find_all('div', class_='card')
#         print("Number of incidents", len(incidents_cards))
        
#         # nothing in this page, skip it -> avoid aving empty months inside the json
#         if len(incidents_cards) == 0: 
#             count_empty_month += 1
#             continue
        
#         incidents_dict[year][month] = []

#         for id_card, card in enumerate(incidents_cards):
#             # print(f"\nCard id : {id_card}")
            
#             # get the different parts of the incident (Title, Resolved, Identified)
#             parts = card.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['status-item', 'ds-width-full'])
            
#             # we don't keep this for now -> and avoid error
#             if len(parts) < 1:
#                 print("No incident or useless one")
#                 break

#             incident_obj = Incident(services=[], identified_datetime=[], resolved_datetime=[]) # force to reset or cache do some shit

#             for id_part, part in enumerate(parts):
#                 part_title = part.find("span", {"class": ["ds-margin-left-12", "ds-text", "ds-font-weight-bold", "ds-color-black"]})

#                 if not part_title:
#                     print("Error, no text inside this part")
#                     continue
                
#                 part_title = part_title.text
                
#                 if id_part == 0: # title of the incident, always the first part
#                     incident_date = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]}).text

#                     incident_obj.title = part_title
#                     incident_obj.card_datetime = extract_word(incident_date.split(), -1, -1) # remove the CEST or other time zone
#                     assign_incident_type(part_title, incident_obj)
#                     continue
                
#                 recover_part_infos(part, part_title, incident_obj)
            
#             incidents_dict[year][month].append(incident_obj)
        
        
#     return count_empty_month

# def scrap_adyen_history(driver, save=True):
#     start_time = time.time()
    
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(script_dir, "data")
#     filename = "adyen_incidents.json"
#     data_file = os.path.join(data_dir, filename)
    
#     incidents_dict = {}
#     try:
#         with open(data_file, 'r') as f:
#             incidents_dict = json.load(f)
            
#     except Exception as e:
#         print("No json file found:", e)

#     today_year = int(datetime.datetime.now().strftime('%Y'))
#     today_month_name = datetime.datetime.now().strftime('%B')
#     id_today_month = list(month_name).index(today_month_name)    

#     if len(incidents_dict) == 0: # never scrapped
#         continue_scrapping = True
#         current_year = today_year
#         while continue_scrapping:
#             count_empty_months = scrap_adyen_months(0, 12, current_year, incidents_dict, driver)
#             if count_empty_months >= 12: # if we have a complete year without any incidents -> we believe that we scrapped everything
#                 continue_scrapping = False
                
#             current_year -= 1
    
#     else:
#         years = incidents_dict.keys()
#         last_year_scrapped = max([int(y) for y in years])
        
#         months = incidents_dict[str(last_year_scrapped)].keys()
#         id_last_month = max([MONTHS.index(m) for m in months])

#         if today_year == last_year_scrapped:
#             scrap_adyen_months(id_last_month, id_today_month, today_year, incidents_dict, driver)
#         elif abs(today_year - last_year_scrapped) == 1:
#             scrap_adyen_months(0, id_today_month, today_year, incidents_dict, driver)
#             scrap_adyen_months(id_last_month, 12, last_year_scrapped, incidents_dict, driver)
#         else:
#             scrap_adyen_months(0, id_today_month, today_year, incidents_dict, driver)
            
#             # get every years between today and last scrapping
#             for i in range(1, abs(today_year - last_year_scrapped)): 
#                 year_to_scrap = today_year - i # compute the year to scrap depending on the today year and the number of years between today and the last scrapping 
#                 scrap_adyen_history(0, 12, year_to_scrap, incidents_dict, driver)
            
#             scrap_adyen_months(id_last_month, 12, last_year_scrapped, incidents_dict, driver)
    
    
#     execution_time = time.time() - start_time
#     # compute the number of element inside the dictionnary
#     incidents_count = 0
#     for year, months in incidents_dict.items():
#         for month, incidents in months.items():
#             incidents_count += len(incidents)
    
#     print(f"""Scrapping finished:
#           - Time taken:                {int(execution_time // 60)} min {int(execution_time % 60)} sec. 
#           - Total number of incidents: {incidents_count}""")

#     if save:
#         print("Starting exporting data into JSON file.")
#         with open(data_file, 'w') as output_file:
#             json.dump(incidents_dict, output_file, indent=4, default=Incident._to_dict)



# ''' Program init '''
# # setup selenium driver
# my_driver = setup_driver(headless=True)
# # Download the NLTK named entity recognition dataset if not already downloaded on the machine
# download_nltk_data()

# scrap_adyen_history(my_driver)


from utils import setup_driver, download_nltk_data
from adyen_scrapper import AdyenScrapper

my_driver = setup_driver()
download_nltk_data()

adyen_scrapper = AdyenScrapper(my_driver)
adyen_scrapper._scrap_adyen_history()