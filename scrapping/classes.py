
class Incident:
    def __init__(self, title=None, service=None, identified_datetime=None, resolved_datetime=None, raw=None,specific_bank=None, card_datetime=None):

        self.title = title
        self.service = service
        self.identified_datetime = identified_datetime
        self.resolved_datetime = resolved_datetime
        self.raw = raw
        self.specific_bank = specific_bank
        self.card_datetime = card_datetime

    def _to_dict(self):
        return {
            "title": self.title,
            "service": self.service,
            "identified_datetime": self.identified_datetime,
            "resolved_datetime": self.resolved_datetime,
            "specific_bank": self.specific_bank,
            "raw": self.raw
        }

    def __str__(self) -> str:
        return ""