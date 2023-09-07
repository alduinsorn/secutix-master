# General import
import json
import os
import time
import re
import datetime
from enum import Enum

# Scrapper module
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# Personal files
from classes import Incident

TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])[:.][0-5][0-9](?:[A-Za-z]{3-4})?\b'
INPUT_DATETIME_FORMAT = '%B %d %Y, %H:%M'
INPUT_TIME_FORMAT = '%H:%M'
OUTPUT_DATETIME_FORMAT = '%Y-%m-%d %H:%M'

class PartType(Enum):
    RESOLVED = 1
    IDENTIFIED = 2
    UPDATED = 3

def setup_driver(headless=True):
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument('--headless')    
    
    driver = webdriver.Firefox(options=options)

    return driver


def parse_url(driver, url, wait_time=2):
    driver.implicitly_wait(wait_time)
    content = None; soup = None

    try:
        driver.get(url)

        try:
            # besoin de cliquer sur un bouton pour afficher tout les incidents
            button_show_all = driver.find_element(By.XPATH, '//button[contains(@class, "ds-button-link ds-button-link--primary ds-button-link--green")]')   

            if button_show_all:
                button_show_all.click()
                time.sleep(wait_time) # sleep require or page isn't complete
        except Exception as e:
            print("No element for this month")
            

        content = driver.page_source
        soup = BeautifulSoup(content, 'lxml')

    except Exception as e:
        print(f"Error: {e}")
        exit()

    return content, soup


def parse_time_in_desc(elem_desc, incident_obj, part_type):
    time_str = re.search(TIME_REGEX, elem_desc)
    if not time_str: print("Error time not found inside the description\n", elem_desc, "\n"); return #input(); return
    time_str = time_str.group().replace('.', ':') # sometimes they have miss typed

    parsed_datetime = datetime.datetime.strptime(incident_obj.card_datetime, INPUT_DATETIME_FORMAT)
    time_var = datetime.datetime.strptime(time_str, INPUT_TIME_FORMAT)
    new_datetime = datetime.datetime(
        year=parsed_datetime.year,
        month=parsed_datetime.month,
        day=parsed_datetime.day,
        hour=time_var.hour,
        minute=time_var.minute
    )

    new_datetime_str = new_datetime.strftime(OUTPUT_DATETIME_FORMAT)

    match part_type:
        case PartType.RESOLVED:
            incident_obj.resolved_datetime = new_datetime_str
        case PartType.IDENTIFIED:
            incident_obj.identified_datetime = new_datetime_str
        case _:
            print("Error: Unknown PartType passed to the function - ", part_type)


def parse_desc_info(elem_desc, incident_obj):
    elem_desc_list = elem_desc.split()
    # contains the service, get all the text between the two words. Sometimes the service contains multiple "words"
    id_through = elem_desc_list.index("through") if "through" in elem_desc_list else -1
    id_starting = elem_desc_list.index("starting") if "starting" in elem_desc_list else -1
    
    if id_through != -1: # if "though" not in the text, it should have "for" and "by"
        elem_service = ' '.join(e for e in elem_desc_list[id_through+1:id_starting])
        incident_obj.service = elem_service
    else:
        # contains the service
        id_for = elem_desc_list.index("for") if "for" in elem_desc_list else -1
        id_transactions = elem_desc_list.index("transactions") if "transactions" in elem_desc_list else -1
        # contains the bank
        id_by = elem_desc_list.index("by") if "by" in elem_desc_list else -1

        elem_service = ' '.join(e for e in elem_desc_list[id_for+1:id_transactions])
        elem_bank = elem_desc_list[id_by+1]

        incident_obj.service = elem_service
        incident_obj.specific_bank = elem_bank  


def scrap_adyen_history(driver):
    # url_history_adyen = "status.adyen.com/incident-history#2023" # only a summary

    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    # months = ["september"]
    years = [2023] # can add more but enough for now
    base_url = "https://status.adyen.com/incident-history"

    incidents_dict = {}

    # go through each pages and get the incidents
    for year in years:
        incidents_dict[year] = {}
        for month in months:
            incidents_dict[year][month] = []
            url = f"{base_url}/{month}-{year}"
            print("URL", url)
            _, soup = parse_url(driver, url)
            
            # contains the incident cards for the month
            incidents_cards = soup.find_all('div', class_='card')
            print("Number of incidents", len(incidents_cards))

            for card in incidents_cards:
                # get the different parts of the incident, contains the Title, Resolved, Identified parts
                parts = card.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['status-item', 'ds-width-full'])
                
                # we don't keep this for now -> and avoid error
                if len(parts) < 1:
                    print("No incident or useless one")
                    break

                incident_obj = Incident()

                for id_part, part in enumerate(parts):
                    part_title = part.find("span", {"class": ["ds-margin-left-12", "ds-text", "ds-font-weight-bold", "ds-color-black"]})

                    if not part_title:
                        print("Error, no text inside this part")
                        continue
                    
                    part_title = part_title.text
                    # print("\npart_title", part_title)
                    
                    match id_part:
                        case 0: # title of the card, the incident
                            incident_date = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]}).text

                            incident_obj.title = part_title
                            incident_obj.card_datetime = ' '.join(incident_date.split()[:-1]) # remove the CEST or other
                        
                        case _: # all the others (Resolved, Identified, Updated)
                            ############### TODO changer cela pour etre proof
                            elem_desc = part.find("p") # ATTENTION CAR PARFOIS L'HEURE N'EST PAS DANS LA PREMIÈRE BALISE <p>
                            
                            if not elem_desc: print(f"Error no <p> in the part. \n{part_title}\n");continue
                            elem_desc = elem_desc.text
                            
                            elem_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]}).text
                            elem_date = ' '.join(elem_date.split()[:-1]) # remove the CEST or other
                            
                            match part_title:
                                case "Identified":
                                    all_elem_desc = part.find_all("p")
                                    raw = ""
                                    for elem in all_elem_desc:
                                        if "Status Page" not in elem.text: 
                                            raw += elem.text
                                    incident_obj.raw = raw
                                    
                                    parse_time_in_desc(elem_desc, incident_obj, PartType.IDENTIFIED)
                                    parse_desc_info(elem_desc, incident_obj)

                                case "Resolved":
                                    parse_time_in_desc(elem_desc, incident_obj, PartType.RESOLVED)
                                case _:
                                    pass # Updated is useless
                                
                
                incidents_dict[year][month].append(incident_obj)
            
    input("Continue...")
    json_output = json.dumps(incidents_dict, indent=4, default=Incident._to_dict)
    print(json_output)

    with open("incidents.json", 'w') as output_file:
        json.dump(incidents_dict, output_file, indent=4, default=Incident._to_dict)


my_driver = setup_driver(True)
scrap_adyen_history(my_driver)
