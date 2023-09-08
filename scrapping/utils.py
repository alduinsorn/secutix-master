from enum import Enum

class Incident:
    def __init__(self, title=None, service=None, identified_datetime=None, resolved_datetime=None, raw="",specific_bank=None, card_datetime=None, incident_type=None):

        self.title = title
        self.service = service
        self.identified_datetime = identified_datetime
        self.resolved_datetime = resolved_datetime
        self.raw = raw
        self.specific_bank = specific_bank
        self.card_datetime = card_datetime
        self.incident_type = incident_type

    def _to_dict(self):
        return {
            "title": self.title,
            "service": self.service,
            "identified_datetime": self.identified_datetime,
            "resolved_datetime": self.resolved_datetime,
            "specific_bank": self.specific_bank,
            "incident_type": self.incident_type,
            "raw": self.raw
        }

    


TIME_REGEX = r'\b(?:[0-1]?[0-9]|2[0-3])[:.][0-5][0-9](?:[A-Za-z]{3-4})?'
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
    REFUSAL_RATE = (1,"Elevated Refusal Rate")
    RESPONSE_TIME = (1,"Response Time") # can be "Elevated Response Time" or "Increase Response Time"
    OFFER_CONVERSION = (2,"Offer Conversion Issues")
    RESOLVED = (3,"[Resolved]") # when the incident is just a report that the saw a problem but it's already solved
    DEGRADED_PERFORMANCE = (4,"Degraded performance")
    OTHER = (5, "")

    def __new__(cls, id, title):
        obj = object.__new__(cls)
        obj._value_ = id
        obj.title = title
        return obj
