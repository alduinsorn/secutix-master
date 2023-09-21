from enum import Enum
from selenium import webdriver


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



##### Generic function ######
def get_word_index(text_list, word):
    return text_list.index(word) if word in text_list else -1

def extract_word(text_list, id_start, id_end):
    return ' '.join(e for e in text_list[id_start+1:id_end])

def setup_driver(headless=True):
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument('--headless')    
    
    driver = webdriver.Firefox(options=options)

    return driver

