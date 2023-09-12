from enum import Enum
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

class Incident:
    def __init__(self, title="Unknown", service=None, identified_datetime=None, resolved_datetime=None, raw="",banks=[], card_datetime=None, incident_type=None):

        self.title = title
        self.service = service
        self.identified_datetime = identified_datetime
        self.resolved_datetime = resolved_datetime
        self.raw = raw
        self.banks = banks
        self.card_datetime = card_datetime
        self.incident_type = incident_type

    def _to_dict(self):
        incident_dict = {
            "title": self.title,
            "service": self.service,
            "identified_datetime": str(self.identified_datetime),
            "resolved_datetime": str(self.resolved_datetime),
            "banks": self.banks,
            "raw": self.raw,
            "card_datetime": self.card_datetime
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
    


TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])\s*[:.]\s*[0-5][0-9](?:[A-Za-z]{3-4})?'
INPUT_DATETIME_FORMAT = '%B %d %Y, %H:%M'
INPUT_TIME_FORMAT = '%H:%M'
OUTPUT_DATETIME_FORMAT = '%Y-%m-%d %H:%M'

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




def download_nltk_data():
    try:
        nltk.data.find('maxent_ne_chunker')
        nltk.data.find('words')
        nltk.data.find('punkt')
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

def extract_proper_names(text):
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