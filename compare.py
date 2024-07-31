"""
Human Computer Interaction Engineering Laboratory
CSAIL, Massachusetts Institute of Technology

Faraz Faruqi
Tarik Hasic
Nayeemur Rahman
Ahmed Katary
x

"""

import pandas as pd
import os
import time
from selenium import webdriver
# from view_helpers import report
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
report = lambda error: f"\033[31m----------------------------\n{error}\n----------------------------\033[0m\n"

###################################################################################################################################
# get old 100 models' ids in a set (SET B) (Should be 90 Things)
# old_data = pd.read_csv('datamodels.csv')

known_ids = set()

# for i in old_data.loc[:, "Link"]:
#     l = i.split("/")
#     try:
#         thing_id = l[3].split(":")[1]
#         known_ids.add(thing_id)
#     except:
#         pass

# print("Known IDS", known_ids)
# print("\n")
###################################################################################################################################

# get new models' ids in a set (SET A) (Faraz told us to find 200 Things)
new_ids = set()

### Global Constants ###
chromedriver = {"path":"/Users/anniepates/Downloads/chromedriver_mac_arm64/chromedriver"}

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
        self.driver = webdriver.Chrome(options=options)

    # scraper to get the new ids
    def scrape_id_getter(self, url, options, parser):
        """
        """
        for option, value in options.items():
            url += f"{option}={value}&"
        print(f"[scraper] >> Scraping {url} ...")

        self.driver.get(url)
        time.sleep(5)
        result = parser(self.driver)
        self.driver.quit()

        return result
    
### Parsers ###
# parser for the new ids
def thingiverse_parser_id_getter(driver):
    """
    This function will get all of the ids of the things on the given Thingiverse page
    """
    links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a') if "ThingCardBody__cardBodyWrapper" in link.get_attribute("class")]
    print(f"[thingiverse_parser] >> found {len(links)} models")

    for link in links:
        l = link.split("/")
        try:
            thing_id = l[3].split(":")[1]
            new_ids.add(thing_id)
        except:
            pass


# parser for the download of images and metadata
def thingiverse_parser_download(driver, download_path, links):
    """
    This function will download everything from the links that are passed in
    The files will be downloaded in the download path
    """
    for link in links:
        try:
            print(f"[thingiverse_parser] >> Fetching {link} ...")
            driver.get(link)
            time.sleep(1)
            
            # Append the name of the Thing to the file_names list
            thing_name = driver.find_element(By.CLASS_NAME, 'ThingTitle__modelName--GN8a6')
            print(thing_name.get_attribute('innerText'))
            file_names.append(thing_name.get_attribute('innerText'))
            
            # download_button = [button for button in driver.find_elements(By.CLASS_NAME, 'Button__blue--1HC2y') if "Button__iconButton" in button.get_attribute("class")][0]
            download_button = driver.find_element(By.CLASS_NAME, 'Button__button--xv8c4')
            download_button.click()
            print(f"[thingiverse_parser] >> {download_button}")
            # If the link is successful, append the link into the success_links lists
            success_links.append(link)
            time.sleep(10)
        except Exception as error: print(f"Error occured while fetching {link}\n{report(error)}")

    # directory for where the new images should be stored
    new_images = "/Users/anniepates/Documents/DREAM/kw_extraction/new_images"
    new_thing_files = "/Users/anniepates/Documents/DREAM/kw_extraction/new_thing_files"

    # for each file, get the images and store them in the new_images directory
    for file in os.listdir(download_path):
        file_name, file_ext = os.path.splitext(file)
        file_path = os.path.join(download_path, file)
        if not os.path.isfile(file_path): continue
        # print("file_name:", file_name, "file_ext:", file_ext, "file_path:", file_path)
        os.system(f"cd {download_path} && unzip -n {file_path} && mv {download_path}/images {download_path}/images_{file_name} && mv {download_path}/images_{file_name} {new_images} && mv {download_path}/files {download_path}/files_{file_name} && mv {download_path}/files_{file_name} {new_thing_files} && cd {download_path} && rm LICENSE.txt && rm README.txt && rm {file_path}")

if __name__ == "__main__":
    # path for where Things should be originally downloaded (directory called "trashy")
    download_path = "/Users/anniepates/Documents/DREAM/kw_extraction/trashy"

    ################################################################################################################################################################
    # First run of WebScraper: Objective: get the IDs of Things that aren't already documented
    start_page = 0
    num_pages = 10
    
    for i in range(start_page, start_page + num_pages):
        options = {
            'page': f"{i + 1}",
            'per_page': "20",
            'sort': "popular",
            'type': "things",
            'q': ""
        }
        scraper = WebScraper(download_path)
        thingiverse_url = "https://www.thingiverse.com/search?"
        result = scraper.scrape_id_getter(thingiverse_url, options, thingiverse_parser_id_getter)
        

    print("New IDS", new_ids)
    print("\n")
    # difference_ids will be the ids that aren't already documented (SET A - SET B)
    difference_ids = new_ids.difference(known_ids)
    print("Difference IDS", difference_ids)
    print("\n")

    print("len known ids", len(known_ids))
    print("len new ids", len(new_ids))
    print("len diff ids", len(difference_ids))

    ################################################################################################################################################################
    # Second run of WebScraper: Objective: Using the new IDs to download their images and store their metadata in a CSV sheet
    links = []
    for thing_id in difference_ids:
        # for each id, generate the link of the Thing and append it to the links list
        links.append("https://www.thingiverse.com/thing:" + thing_id)

    # initialize variables to store into CSV files
    file_names = []
    success_links = []
    list_of_ids = []
    
    scraper = WebScraper(download_path)
    result = thingiverse_parser_download(scraper.driver, download_path, links)

    for link in success_links:
        s = link.split("/")
        l = s[-1].split(":")
        list_of_ids.append(l[1])

    print("File Names:", len(file_names), "Links:", len(success_links), "IDs:", len(list_of_ids))
    # for each row in the csv file, create columns for Name, Link, ID, Class, Components, and Functional Segments
    d = {'Name': file_names, 'Link': success_links, 'ID': list_of_ids, 'Class': ["" for i in range(len(file_names))], 'Components': ["" for i in range(len(file_names))], 'Functional Segments': ["" for i in range(len(file_names))]}
    df = pd.DataFrame(data=d)
    # alphabatize the df by "Name" column
    sorted_df = df.sort_values(by=["Name"])
    print(sorted_df)
    # turn the sorted df into a csv file
    sorted_df.to_csv("new_models.csv")
