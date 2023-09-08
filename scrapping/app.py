# General import
import json
import os
import time
import re
import datetime

# Scrapper module
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# Personal files
from utils import *


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


def find_right_tag_containing_time(elem_desc_all, incident_obj, part_type):
    for desc in elem_desc_all:
        if parse_time_in_desc(desc.text, incident_obj, part_type):
            return

    print("Error no time in every text part")

def parse_time_in_desc(elem_desc, incident_obj, part_type):
    time_str = re.search(TIME_REGEX, elem_desc)
    if not time_str: print("Error time not found inside the description\n", elem_desc, "\n"); return False
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
        case PartType.RESOLVED.value:
            incident_obj.resolved_datetime = new_datetime_str
        case PartType.IDENTIFIED.value:
            incident_obj.identified_datetime = new_datetime_str
        case _:
            print("Error: Unknown PartType passed to the function - ", part_type)


def retrieve_error_rate(elem_desc_list, incident_obj):
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

        # contains the bank(s) -> can have multiple bank specified
        id_by = elem_desc_list.index("by") if "by" in elem_desc_list else -1

        elem_service = ' '.join(e for e in elem_desc_list[id_for+1:id_transactions])
        elem_bank = elem_desc_list[id_by+1]

        incident_obj.service = elem_service
        incident_obj.specific_bank = elem_bank  
    

def retrieve_offer_conversion():
    
    pass


def parse_desc_info(elem_desc_all, incident_obj):
    return
    for elem_desc in elem_desc_all[:1]:        
        elem_desc_list = elem_desc.split()

        match incident_obj.incident_type.value:
            case IncidentType.ERROR_RATE.value: # this also take the REFUSAL_RATE and RESPONSE_TIME
                retrieve_error_rate(elem_desc_list, incident_obj)
            case IncidentType.OFFER_CONVERSION.value:
                pass
            case IncidentType.RESOLVED.value:
                pass
            case IncidentType.DEGRADED_PERFORMANCE.value:
                pass
            case IncidentType.OTHER.value:
                pass

    


def check_incident_type(part_title, incident_obj):
    part_title_lowered = part_title.lower()
    incident_obj.incident_type = IncidentType.OTHER # by default we set the Incident as OTHER

    for i in IncidentType:
        if i.title.lower() in part_title_lowered:
            incident_obj.incident_type = i 


def retrieve_raw_desc(part, incident_obj, part_type):
    all_elem_desc = part.find_all("p")
    raw = f"{part_type.title}("
    for elem in all_elem_desc:
        if "https://status.adyen.com" not in elem.text: 
            raw += elem.text
    raw += ")\n"
    incident_obj.raw += raw


def scrap_adyen_history(driver, save=False):
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

            for id_card, card in enumerate(incidents_cards):
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
                    
                    # part_title can contains : ["title of the incident", "resolved", "identified", "updated"]
                    part_title = part_title.text
                    # print("\npart_title", part_title)
                    check_incident_type(part_title, incident_obj)
                    
                    match id_part:
                        case PartType.TITLE.value: # title of the card, the incident
                            incident_date = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]}).text

                            incident_obj.title = part_title
                            incident_obj.card_datetime = ' '.join(incident_date.split()[:-1]) # remove the CEST or other
                        
                        case PartType.IDENTIFIED.value: # all the others (Resolved, Identified, Updated)
                            elem_desc_all = part.find_all("p") # we get all the <p> balises in the case that the informations aren't in the first <p>

                            if len(elem_desc_all) == 0: # sometimes the Resolved part has no text -> no an error
                                print(f"No <p> in the part: {part_title} - card_id : {id_card}\n")
                                continue

                            elem_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]}).text
                            elem_date = ' '.join(elem_date.split()[:-1]) # remove the CEST or other
                            
                            retrieve_raw_desc(part, incident_obj, PartType.IDENTIFIED)
                            find_right_tag_containing_time(elem_desc_all, incident_obj, PartType.IDENTIFIED.value)
                            parse_desc_info(elem_desc_all, incident_obj)

                        case PartType.RESOLVED.value:
                            elem_desc = part.find("p")

                            if not elem_desc:
                                print(f"No <p> in the part Resolved. \n{part_title} - card_id : {id_card}\n")
                                continue
                            
                            elem_desc = elem_desc.text

                            parse_time_in_desc(elem_desc, incident_obj, PartType.RESOLVED.value)
                            retrieve_raw_desc(part, incident_obj, PartType.RESOLVED)
                            
                        case _: pass # Updated not taken into account
                
                incidents_dict[year][month].append(incident_obj)
            
    input("Continue...")
    json_output = json.dumps(incidents_dict, indent=4, default=Incident._to_dict)
    print(json_output)

    if save:
        with open("incidents.json", 'w') as output_file:
            json.dump(incidents_dict, output_file, indent=4, default=Incident._to_dict)


my_driver = setup_driver(True)
scrap_adyen_history(my_driver)
