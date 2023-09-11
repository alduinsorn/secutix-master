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

    
def assign_incident_type(part_title, incident_obj):
    part_title_lowered = part_title.lower()
    incident_obj.incident_type = IncidentType.OTHER # by default we set the Incident as OTHER

    for i in IncidentType:
        if i.title.lower() in part_title_lowered:
            incident_obj.incident_type = i
            return

def retrieve_raw_desc(part, incident_obj, part_title):
    all_elem_desc = part.find_all("p")
    raw = f"{part_title}("
    for elem in all_elem_desc:
        if "https://status.adyen.com" not in elem.text: 
            raw += elem.text
    raw += ")\n"
    incident_obj.raw += raw
    # print("raw", raw)

def search_assign_datetime(elem_desc_all, elem_date, incident_obj, part_title):
    for desc in elem_desc_all:
        time_str = re.search(TIME_REGEX, desc.text)
        if not time_str: continue # search in the different <p> tags
        time_str = time_str.group().replace('.', ':').replace(" ", "") # sometimes they have miss typed

        print("search_assign_datetime -> time found")

        # Datetime normalization 
        ''' TODO 
        TRY CATCH REQUIRED
        '''
        parsed_datetime = datetime.datetime.strptime(elem_date, INPUT_DATETIME_FORMAT)
        time_var = datetime.datetime.strptime(time_str, INPUT_TIME_FORMAT)
        
        new_datetime = datetime.datetime(
            year=parsed_datetime.year,
            month=parsed_datetime.month,
            day=parsed_datetime.day,
            hour=time_var.hour,
            minute=time_var.minute
        )
        new_datetime_str = new_datetime.strftime(OUTPUT_DATETIME_FORMAT)

        match part_title:
            case PartType.RESOLVED.title:
                incident_obj.resolved_datetime = new_datetime_str
            case PartType.IDENTIFIED.title:
                incident_obj.identified_datetime = new_datetime_str
            case _:
                print("Error: Unknown PartType passed to the function - ", part_title)
        return True

    print("Error no time in every <p> balise")

'''TODO finish the code in this part'''
def search_date_in_desc(elem_desc_all, incident_obj):
    # search for the date in text -> can vary a lot
    return False

def get_word_index(text_list, word):
    return text_list.index(word) if word in text_list else -1

def extract_word(text_list, id_start, id_end):
    return ' '.join(e for e in text_list[id_start+1:id_end])

def process_classic_case(id, desc_word_list, incident_obj, word):
    id_word = get_word_index(desc_word_list, word)
    if id_word >= 0:
        elem_service = extract_word(desc_word_list, id, id_word)
        incident_obj.service = elem_service

def process_for_by_case(id_1, id_2, desc_word_list, incident_obj):
    id_transactions = get_word_index(desc_word_list, "transactions")
    if id_transactions >= 0:
        incident_obj.service = extract_word(desc_word_list, id_2, id_transactions)
    
    id_account = get_word_index(desc_word_list, "account")
    id_cardholders = get_word_index(desc_word_list, "cardholders")
    if id_account >= 0 and id_cardholders >= 0:
        id_end = id_account if id_account >= 0 else id_cardholders
        incident_obj.banks.append(extract_word(desc_word_list, id_1, id_end))

def search_service_n_bank(elem_desc_all, incident_obj):
    keyword_actions = {
        "through": process_classic_case,
        "for_by": process_for_by_case,
        "with": process_classic_case
    }
            
    for i, elem_desc in enumerate(elem_desc_all):
        desc_word_list = elem_desc.text.split()
        
        # cas 1 contient mot clé "through"
        id_through = get_word_index(desc_word_list, "through")
        
        # cas 2 contient mot clé "for" et "by" ce qui fournit le service et la banque
        id_for = get_word_index(desc_word_list, "for")
        id_by = get_word_index(elem_desc_all, "by")
        
        # cas 3 contient mot clé "with"
        id_with = get_word_index(elem_desc_all, "with")
        
        if id_through != -1 and "through" in keyword_actions:
            keyword_actions['through'](id_through, desc_word_list, incident_obj, "through")
        elif id_for != -1 and id_by != -1 and "for_by" in keyword_actions:
            keyword_actions['for_by'](id_for, id_by, desc_word_list, incident_obj)
        if id_with != -1 and "with" in keyword_actions:
            keyword_actions['with'](id_with, desc_word_list, incident_obj, "with")
        
        # Add all proper names in the bank arrray -> should do everything -> service and bank
        proper_names = extract_proper_names(elem_desc.text)
        for bank in proper_names:
            if bank.strip() != incident_obj.service:
                incident_obj.banks.append(bank.strip())
        incident_obj.banks = list(dict.fromkeys(incident_obj.banks))
        
def recover_part_infos(part, part_title, incident_obj):
    elem_desc_all = part.find_all("p") 
    if len(elem_desc_all) == 0: # sometimes the Resolved part has no text -> not an error
        print(f"No <p> in the part: {part_title}\n")
        return

    
    # Check if the date is written in the description or take the part time
    elem_date = search_date_in_desc(elem_desc_all, incident_obj)
    print("elem_date :", elem_date)
    if not elem_date:
        elem_date = part.find("span", {"class": ["ds-text-small ds-color-grey-450 ds-margin-left-12"]}).text
        elem_date = ' '.join(elem_date.split()[:-1]) # remove the CEST or other

    print("elem_date :", elem_date)

    search_assign_datetime(elem_desc_all, elem_date, incident_obj, part_title)
    retrieve_raw_desc(part, incident_obj, part_title)
    
    match part_title:
        case PartType.IDENTIFIED.title:
            search_service_n_bank(elem_desc_all, incident_obj)



def retrieve_offer_conversion():
    
    pass



# Main function deciding of what we gonna do with the given part
def process_incident_details(part, part_title, incident_obj):
    if incident_obj.incident_type.value in [IncidentType.ERROR_RATE.value, IncidentType.REFUSAL_RATE.value, IncidentType.RESPONSE_TIME.value]:
        recover_part_infos(part, part_title, incident_obj)            
    elif incident_obj.incident_type.value == IncidentType.OFFER_CONVERSION.value:
        pass
    elif incident_obj.incident_type.value == IncidentType.RESOLVED.value:
        pass
    elif incident_obj.incident_type.value == IncidentType.DEGRADED_PERFORMANCE.value:
        pass
    elif incident_obj.incident_type.value == IncidentType.OTHER.value:
        pass


def scrap_adyen_history(driver, save=False):
    # url_history_adyen = "status.adyen.com/incident-history#2023" # only a summary

    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    # months = ["september"]
    months = ["january"]
    
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

            for id_card, card in enumerate(incidents_cards[:5]):
                print(f"\n\nCard id : {id_card}")
                
                # get the different parts of the incident, contains the Title, Resolved, Identified parts
                parts = card.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['status-item', 'ds-width-full'])
                
                # we don't keep this for now -> and avoid error
                if len(parts) < 1:
                    print("No incident or useless one")
                    break

                incident_obj = Incident(banks=[])
                print(incident_obj)

                for id_part, part in enumerate(parts):
                    part_title = part.find("span", {"class": ["ds-margin-left-12", "ds-text", "ds-font-weight-bold", "ds-color-black"]})

                    if not part_title:
                        print("Error, no text inside this part")
                        continue
                    
                    # part_title can contains : ["title of the incident", "resolved", "identified", "updated"]
                    part_title = part_title.text
                    
                    if id_part == 0: # always true - # title of the card, the incident
                        incident_date = part.find("span", {"class": ["ds-text-small", "ds-color-grey-450"]}).text

                        incident_obj.title = part_title
                        incident_obj.card_datetime = ' '.join(incident_date.split()[:-1]) # remove the CEST or other
                        assign_incident_type(part_title, incident_obj)

                        # print("After title :", str(incident_obj))
                        continue
                    
                    process_incident_details(part, part_title, incident_obj)
                print(incident_obj)
                incidents_dict[year][month].append(incident_obj)
            
    input("Continue...")
    json_output = json.dumps(incidents_dict, indent=4, default=Incident._to_dict)
    print(json_output)

    if save:
        with open("incidents.json", 'w') as output_file:
            json.dump(incidents_dict, output_file, indent=4, default=Incident._to_dict)



''' Program init '''
# setup selenium driver
my_driver = setup_driver(True)
# Download the NLTK named entity recognition dataset if not already downloaded
download_nltk_data()
    


scrap_adyen_history(my_driver, True)
