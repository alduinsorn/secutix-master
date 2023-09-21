from enum import Enum
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tokenize import word_tokenize

from time import sleep

# Scrapper module
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

import datetime
import re
from dateutil.parser import parse


TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])\s*[:.]\s*[0-5][0-9](?:[A-Za-z]{3-4})?'
DATE_REGEX_1 = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*\b'
DATE_REGEX_2 = r'\b\d{1,2}(?:st|nd|rd|th)?\s*(?:of)?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'

INPUT_DATETIME_FORMAT = '%B %d %Y, %H:%M'
INPUT_TIME_FORMAT = '%H:%M'
OUTPUT_DATETIME_FORMAT = '%Y-%m-%d %H:%M'

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]



class Incident:
    def __init__(self, title="Unknown", services=[], identified_datetime=[], resolved_datetime=[], raw="", card_datetime=None, incident_type=None):

        self.title = title
        self.services = services
        self.identified_datetime = identified_datetime
        self.resolved_datetime = resolved_datetime
        self.raw = raw
        self.card_datetime = card_datetime
        self.incident_type = incident_type

    def _to_dict(self):
        incident_dict = {
            "title": self.title,
            "services": self.services,
            "identified_datetime": self.identified_datetime,
            "resolved_datetime": self.resolved_datetime,
            "raw": self.raw,
            "card_datetime": str(self.card_datetime)
        }
        if self.incident_type:
            incident_dict["incident_type"] = {
                "id" : self.incident_type.value,
                "title" : self.incident_type.title
            }
        return incident_dict
    
    
    def __str__(self):
        attributes = self._to_dict()
        json_str = "{\n"
        for key, value in attributes.items():
            json_str += f'    "{key}": "{value}",\n'
        json_str += "}"
        return json_str
    

class PartType(Enum):
    TITLE = (0, "Title")
    RESOLVED = (1, "Resolved")
    IDENTIFIED = (2, "Identified")
    UPDATED = (3, "Updated")

    def __new__(cls, id, title):
        obj = object.__new__(cls)
        obj._value_ = id
        obj.title = title
        return obj


class IncidentType(Enum):
    ERROR_RATE = (1,"Elevated Error Rate")
    REFUSAL_RATE = (2,"Elevated Refusal Rate")
    RESPONSE_TIME = (3,"Response Time") # can be "Elevated Response Time" or "Increase Response Time"
    OFFER_CONVERSION = (4,"Offer Conversion Issues")
    RESOLVED = (5,"[Resolved]") # when the incident is just a report that the saw a problem but it's already solved
    DEGRADED_PERFORMANCE = (6,"Degraded performance")
    OTHER = (7, "Other")

    def __new__(cls, id, title):
        obj = object.__new__(cls)
        obj._value_ = id
        obj.title = title
        return obj

    def get_incident_type(part_title):
        part_title_lowered = part_title.lower()
        
        for i in IncidentType:
            if i.title.lower() in part_title_lowered:
                return i
        return IncidentType.OTHER



class DatetimeUtils:
    # Function that search any times given in the description like "12:15 CEST" and then assign it to the correct datetime attribut
    def search_times(elem_desc_all, elem_date):
        # General datetime from the part
        general_parsed_datetime = elem_date if isinstance(elem_date, datetime.datetime) else datetime.datetime.strptime(elem_date, INPUT_DATETIME_FORMAT)
        datetime_arr = []
        
        # For every <p> tag in the part description
        for desc in elem_desc_all:
            times_arr = re.findall(TIME_REGEX, desc.text) # get all the time
            for time_str in times_arr:
                time_found = DatetimeUtils.clean_time(time_str, general_parsed_datetime)
                if time_found:
                    datetime_arr.append(time_found)

        datetime_arr = list(set(datetime_arr))
        return [datetime.datetime.strftime(general_parsed_datetime, INPUT_DATETIME_FORMAT)] if len(datetime_arr) == 0 else datetime_arr

    def clean_time(time_str, general_parsed_datetime):
        if not time_str: return None
        time_str = time_str.replace('.', ':').replace(" ", "") # sometimes they have miss typed

        time_var = datetime.datetime.strptime(time_str, INPUT_TIME_FORMAT)
        
        new_datetime = datetime.datetime(
            year = general_parsed_datetime.year,
            month = general_parsed_datetime.month,
            day = general_parsed_datetime.day,
            hour = time_var.hour,
            minute = time_var.minute
        )
        return new_datetime.strftime(OUTPUT_DATETIME_FORMAT)

    def search_dates(elem_desc_all):
        parsed_dates = []
        for elem_desc in elem_desc_all:
            # search for the date in text -> can vary a lot
            date_matches = re.findall(DATE_REGEX_1, elem_desc.text, re.IGNORECASE)
            date_matches.extend(re.findall(DATE_REGEX_2, elem_desc.text, re.IGNORECASE))
            for date_match in date_matches:
                parsed_date = parse(date_match, fuzzy=True)
                parsed_dates.append(parsed_date)
        
        return parsed_dates

    def convert_to_date(text, date_format):
        return datetime.datetime.strptime(text, date_format)

##### NLP Part #####

def download_nltk_data():
    try:
        nltk.data.find('maxent_ne_chunker')
        nltk.data.find('words')
        nltk.data.find('punkt')
        nltk.data.find('averaged_perceptron_tagger')
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

# Extract the minimum
def extract_proper_names_nltk(text):
    # Tokenize the words into sentences
    proper_names = []
    
    nltk_results = ne_chunk(pos_tag(word_tokenize(text[1:-1])))
    for nltk_res in nltk_results:
        if type(nltk_res) == Tree:
            name = ''
            for nltk_res_leaf in nltk_res.leaves():
                name += nltk_res_leaf[0] + ' '
            proper_names.append(name)
    
    return proper_names

def extract_proper_names_spacy(text):
    import spacy
    import time
    # nlp = spacy.load("en_core_web_sm") # 5min pour scrapper
    nlp = spacy.load("en_core_web_trf") # 30 min pour scrapper -> il faut changer la logique de la fonction search_services() pour envoyer un texte plus long pour que nlp soit appeler 1x avec tout le texte

    # start_time = time.time()
    proper_names = []
    
    doc = nlp(text)
    
    for entity in doc.ents:
        if entity.label_ in ["PERSON", "ORG"]:
            proper_names.append(entity.text)

    # print(f"Proper names extracted in {time.time() - start_time} seconds")
    return proper_names



import re
import nltk
from nltk.corpus import stopwords

# Assurez-vous d'avoir téléchargé la liste des stopwords
nltk.download('stopwords')

# Fonction pour extraire les noms propres fait maison
def extract_proper_names(text):
    proper_names = []
    
    # Tokenize text
    words = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))  
    
    prev_word_was_upper = False
    current_name = ""

    for word in words:
        if any(letter.isupper() for letter in word) and word.lower() not in stop_words:
            # allow to get the proper noun if it has multiple word in it
            if not prev_word_was_upper:
                current_name = word
            else:
                current_name += ' ' + word
            prev_word_was_upper = True
        else:
            if current_name:
                proper_names.append(current_name)
                current_name = ""
            prev_word_was_upper = False
    
    # Add the last proper noun found (combined)
    if current_name:
        proper_names.append(current_name)
    
    return proper_names



##### Generic function ######

def get_word_index(text_list, word):
    return text_list.index(word) if word in text_list else -1

def extract_word(text_list, id_start, id_end):
    return ' '.join(e for e in text_list[id_start+1:id_end])



##### General scrapper functions #####

def setup_driver(headless=True):
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument('--headless')    
    
    driver = webdriver.Firefox(options=options)

    return driver


def clean_services_found(services, words_often_found):
    cleaned_services = []
    seen_services = set()
    
    for service in services:
        lowercase_service = service.lower()
        if lowercase_service not in seen_services and service not in words_often_found:
            cleaned_services.append(service)
            seen_services.add(lowercase_service)
    
    return cleaned_services


def search_services(elem_desc_all: list):
    services = []
    for elem_desc in elem_desc_all:
        # Add all proper names in the bank arrray -> should do everything -> service and bank
        proper_names = extract_proper_names(elem_desc.text)
        for name in proper_names:
            services.append(name.strip())

    return services