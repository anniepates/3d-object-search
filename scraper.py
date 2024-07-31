"""
Human Computer Interaction Engineering Laboratory
CSAIL, Massachusetts Institute of Technology

Faraz Faruqi
Tarik Hasic
Nayeemur Rahman
Ahmed Katary


"""

import os

import time
from selenium import webdriver
#from view_helpers import report
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

### Global Constants ###
chromedriver = "/home/ubuntu/chromedriver"
report = lambda error: f"\033[31m----------------------------\n{error}\n----------------------------\033[0m\n"


class WebScraper():
    """
    """
    def __init__(self, download_path):

        options = Options()
        options.headless = True
        options.add_argument("--window-size=1920,1200")
        
        prefs = {"download.default_directory" : download_path}
        options.add_experimental_option("prefs",prefs)

        self.download_path = download_path
        self.driver = webdriver.Chrome(options=options) #, executable_path=chromedriver

    def scrape(self, url, options, parser):
        """
        """
        for option, value in options.items():
            url += f"{option}={value}&"
        print(f"[scraper] >> Scraping {url} ...")

        self.driver.get(url)
        time.sleep(5)
        result = parser(self.driver, self.download_path)
        self.driver.quit()

        return result
    
### Parsers ###
def thingiverse_parser(driver, download_path):
    """
    """
    links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a') if "ThingCardBody__cardBodyWrapper" in link.get_attribute("class")]
    print(f"[thingiverse_parser] >> found {len(links)} models")

    for link in links:
        try:
            print(f"[thingiverse_parser] >> Fetching {link} ...")
            driver.get(link)
            download_button = [button for button in driver.find_elements(By.CLASS_NAME, 'Button__button--xv8c4') if "Button__icon--UCU2X" in button.get_attribute("class")][0]
            download_button.click()
            print(f"[thingiverse_parser] >> {download_button}")
            time.sleep(10)
        except Exception as error: print(f"Error occured while fetching {link}\n{report(error)}")
    
    for file in os.listdir(download_path):
        file_name, file_ext = os.path.splitext(file)
        file_path = os.path.join(download_path, file)
        if not os.path.isfile(file_path): continue
        os.system(f"mkdir {download_path}/{file_name} && cd {download_path}/{file_name} && unzip {file_path} && mv {file_path} {download_path}/{file_name}")
        
# # Parser to get Title + Description
# def title_parser(driver, download_path):
#     a=1 


if __name__ == "__main__":
    start_page = 5
    num_pages = 1
    for i in range(start_page, start_page + num_pages):
        options = {
            'page': f"{i + 1}",
            'per_page': "1",
            'sort': "popular",
            'posted_after': "now-30d",
            'type': "things",
            'q': "mug"
        }
        download_path = "/Users/anniepates/Documents/DREAM/kw_extraction/pill_box_meshes"

        scraper = WebScraper(download_path)
        thingiverse_url = "https://www.thingiverse.com/search?"
        result = scraper.scrape(thingiverse_url, options, thingiverse_parser)