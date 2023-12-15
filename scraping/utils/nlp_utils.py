# from nltk import data as nltk_data, download as nltk_download
# from nltk.corpus import stopwords
from re import findall as re_findall
from typing import List, Set
import spacy

# def download_nltk_data() -> None:
#     '''
#     This function download the nltk data if not already downloaded
    
#     Returns:
#         None
#     '''
#     try:
#         nltk_data.find('stopwords')
#     except LookupError:
#         nltk_download('stopwords')


# # Fonction pour extraire les noms propres fait maison
# def extract_proper_names(text: str) -> List[str]:
#     '''
#     This function extract the "proper names" contained in the text (based on the capital letters)
    
#     Parameters:
#         text (str): text to extract the proper names from
        
#     Returns:
#         list<str>: list of all the proper names found
#     '''
#     proper_names: List[str] = []
    
#     # Before tokenizing, convert '\n' to '.'
#     text = text.replace('\n', '. ')

#     # Tokenize text
#     words: List[str] = re_findall(r'\b\w+\b', text)
#     stop_words: Set[str] = set(stopwords.words('english'))  
    
#     prev_word_was_upper: bool = False
#     current_name: str = ""

#     for word in words:
#         if any(letter.isupper() for letter in word) and word.lower() not in stop_words:
#             # allow to get the proper noun if it has multiple word in it
#             if not prev_word_was_upper:
#                 current_name = word
#             else:
#                 current_name += ' ' + word
#             prev_word_was_upper = True
#         else:
#             if current_name:
#                 proper_names.append(current_name)
#                 current_name = ""
#             prev_word_was_upper = False
    
#     # Add the last proper noun found (combined)
#     if current_name:
#         proper_names.append(current_name)
    
#     return proper_names

class NLPUtils:
    def __init__(self, psp_name):
        self.nlp = spacy.load("en_core_web_sm")
        self.psp_name = psp_name
        self.payment_methods_names: List[str] = self._load_payment_methods()

    def _load_payment_methods(self) -> List[str]:
        with open(f"utils/payment_methods_{self.psp_name}.txt", "r") as f:
            payment_methods: List[str] = f.readlines()
        payment_methods = [method.replace("\n", "") for method in payment_methods]
        return payment_methods

    def _extract_proper_names_spacy(self, text: str) -> List[str]:
        '''
        This function extract the "proper names" contained in the text (based on the capital letters)
        
        Parameters:
            text (str): text to extract the proper names from
            
        Returns:
            list<str>: list of all the proper names found
        '''
        # print("before", text)
        # text = text.replace('.', '. ').replace(',', ', ').replace('\n', ' ')
        # print("after",text)

        # extract entities from the text and keep only the ones that are ORG
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        specific_entities = [entity[0] for entity in entities if entity[1] == "ORG"]

        # in the text, extract words that are completely capitalized
        full_capitalized_words: List[str] = [word for word in text.split() if word.isupper()]
        specific_entities.extend(full_capitalized_words)
        
        # specific case for some words that are often found in the description but don't mean anything
        for specific in ["CET", "CEST", "CET.", "CEST.", "CET,", "CEST,", "RESOLVED", "COMPLETED", "IDENTIFIED", "PRODUCTION", "TEST", "ScheduledWorldline", "Please"]:
            if specific in specific_entities:
                while specific in specific_entities:
                    specific_entities.remove(specific)
            for idx_specific, each_specific in enumerate(specific_entities):
                if specific in each_specific:
                    specific_entities[idx_specific] = specific_entities[idx_specific].replace(specific, "")
        
        # Keep words that are longer than 2 characters
        specific_entities = [word for word in specific_entities if len(word) > 2]

        # remove special characters
        for i in range(len(specific_entities)):
            specific_entities[i] = specific_entities[i].replace(")", "").replace("(", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("\n", "")

        return list(set(specific_entities))

    def _clean_services_found(self, services: List[str]) -> List[str]:
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
        
        # avoid duplicates
        for service in services:
            lowercase_service: str = service.lower()
            if lowercase_service not in seen_services:
                # check if the lowercase_service is not a substring of another service
                # if not any(lowercase_service in seen_service for seen_service in seen_services):
                cleaned_services.append(service)
                seen_services.add(lowercase_service)           
        
        return cleaned_services

    def _search_payment_methods(self, elem_desc_all: List[str]) -> List[str]:
        payment_methods: List[str] = []
        for elem_desc in elem_desc_all:
            elem_desc = elem_desc.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ").replace("\n", " ")
            for method in self.payment_methods_names:
                if method in elem_desc:
                    payment_methods.append(method)
        return payment_methods

    def _search_services(self, elem_desc_all: List[str]) -> List[str]:
        '''
        This function is a wrapper for the extract_proper_names and clean_services_found functions
        
        Parameters:
            elem_desc_all (List[str]): list of text (description of the incident)
            words_often_found (List[str]): list of words often found in the description

        Returns:
            List[str]: list of all the services found in the description
        '''
        # extract the proper names from the description
        services: List[str] = []
        for elem_desc in elem_desc_all:
            proper_names: List[str] = self._extract_proper_names_spacy(elem_desc)
            for name in proper_names:
                services.append(name)

        # clean the services found
        cleaned_services: List[str] = self._clean_services_found(services)

        payment_methods: List[str] = self._search_payment_methods(elem_desc_all)
        cleaned_services = [service for service in cleaned_services if service not in payment_methods]

        return cleaned_services, payment_methods