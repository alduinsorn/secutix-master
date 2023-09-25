from nltk import data as nltk_data, download as nltk_download
from nltk.corpus import stopwords
from re import findall as re_findall

from typing import List, Set

def download_nltk_data() -> None:
    '''
    This function download the nltk data if not already downloaded
    
    Returns:
        None
    '''
    try:
        nltk_data.find('stopwords')
    except LookupError:
        nltk_download('stopwords')


# Fonction pour extraire les noms propres fait maison
def extract_proper_names(text: str) -> List[str]:
    '''
    This function extract the "proper names" contained in the text (based on the capital letters)
    
    Parameters:
        text (str): text to extract the proper names from
        
    Returns:
        list<str>: list of all the proper names found
    '''
    proper_names: List[str] = []
    
    # Tokenize text
    words: List[str] = re_findall(r'\b\w+\b', text)
    stop_words: Set[str] = set(stopwords.words('english'))  
    
    prev_word_was_upper: bool = False
    current_name: str = ""

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


def clean_services_found(services: List[str], words_often_found: List[str]) -> List[str]:
    '''
    This function clean the services found by removing the duplicates and the words often found in the description (words_often_found)
    
    Parameters:
        services (list<str>): list of services found
        words_often_found (list<str>): list of words often found in the descriptionm
        
    Returns:
        list<str>: list of cleaned services
    '''
    cleaned_services: List[str] = []
    seen_services: Set[str] = set()
    
    for service in services:
        lowercase_service: str = service.lower()
        if lowercase_service not in seen_services and service not in words_often_found:
            cleaned_services.append(service)
            seen_services.add(lowercase_service)
    
    return cleaned_services


def search_services(elem_desc_all: List[str], words_often_found: List[str]):
    '''
    This function is a wrapper for the extract_proper_names and clean_services_found functions
    
    Parameters:
        elem_desc_all (list<str>): list of text (description of the incident)
        words_often_found (list<str>): list of words often found in the description

    Returns:
        list<str>: list of all the services found in the description
    '''
    services: List[str] = []
    for elem_desc in elem_desc_all:
        proper_names: List[str] = extract_proper_names(elem_desc.text)
        for name in proper_names:
            services.append(name.strip())

    cleaned_services: List[str] = clean_services_found(services, words_often_found)

    return cleaned_services