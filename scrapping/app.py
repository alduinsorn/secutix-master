from utils.utils import setup_driver
from utils.nlp_utils import download_nltk_data
from adyen_scrapper import AdyenScrapper

my_driver = setup_driver()
download_nltk_data()

adyen_scrapper = AdyenScrapper(my_driver)
adyen_scrapper._scrap_adyen_history()