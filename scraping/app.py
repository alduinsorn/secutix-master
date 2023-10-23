from utils.utils import setup_driver
from utils.nlp_utils import download_nltk_data
from adyen_scrapper import AdyenScrapper
from ingenico_scrapper import IngenicoScrapper

import psutil
import time


process = psutil.Process()

my_driver = setup_driver(headless=True)
download_nltk_data()

adyen_scrapper = AdyenScrapper(my_driver, json_export=True)
adyen_scrapper._scrap_adyen_history()

# # memory_usage = process.memory_info()
# # memory_mb = memory_usage.rss / (1024 ** 2)  # Convert to megabytes
# # print(f"Memory Usage: {memory_mb:.2f} MB")

# import sys

# # Calcule la taille de l'objet en octets
# dict_size_bytes = sys.getsizeof(adyen_scrapper.incidents_dict)

# # Convertit la taille en mégaoctets (MB)
# dict_size_mb = dict_size_bytes / (1024 ** 2)  # 1 MB = 1024 * 1024 octets
# print(f"dict size {dict_size_mb} MB")



# ingenico_scrapper = IngenicoScrapper(my_driver, json_export=False)
# ingenico_scrapper._scrap_history()