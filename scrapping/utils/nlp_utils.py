from nltk import data as nltk_data, download as nltk_download
from nltk.corpus import stopwords
from re import findall as re_findall

def download_nltk_data():
    try:
        nltk_data.find('stopwords')
    except LookupError:
        nltk_download('stopwords')


# Fonction pour extraire les noms propres fait maison
def extract_proper_names(text):
    proper_names = []
    
    # Tokenize text
    words = re_findall(r'\b\w+\b', text)
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