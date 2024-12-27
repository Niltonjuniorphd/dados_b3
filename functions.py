from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def call_driver():
    """
    Initializes and returns a headless Chrome WebDriver instance.
    If the driver is not found, automatically install the appropriate version of the ChromeDriver.

    Returns:
        WebDriver: An instance of the Chrome WebDriver configured to run headless.
    """
    options = Options()
    # options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    options.add_argument("--incognito")  # Run in incognito mode
    options.add_argument("--no-sandbox")  # Disable sandboxing
    # Overcome limited resource problems
    options.add_argument("--disable-dev-shm-usage")
    # Set a user-agent string
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36")

    # Initialize the driver with the specified options
    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)
    print("Driver ready... ")

    return driver


def send_focus_key(driver, focus_key):
    """
    Sends a search query to Google using the provided WebDriver instance.

    This function navigates to the Google homepage, waits for the search box
    to become visible, and then sends the specified `focus_key` as a search query.
    After submitting the query, it waits briefly to allow the page to load.

    Args:
        driver (WebDriver): The WebDriver instance used to interact with the browser.
        focus_key (str): The search query to be sent to Google's search box.

    Raises:
        Exception: If any error occurs during navigation or interaction with the search
        box.

    """
    try:
        driver.get("https://google.com/ncr")
        WebDriverWait(driver, 5).until(EC.url_contains("google.com"))
        search_box = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "q")))
        print(f"Focus key: -{focus_key}- sent to google serach box... ")
        search_box.clear()  # Clear any pre-existing text
        search_box.send_keys(focus_key)
        search_box.send_keys(Keys.RETURN)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "search")))

    except Exception as e:
        print(f"Failed to send focus key: {e}")


def get_text_and_links(driver, pg_num=20):
    links = []
    texts = []
    headlines = []
    k = 0
    count_elements = 0

    for s in range(pg_num):
        try:
            elements = WebDriverWait(driver, 4).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='MjjYud']")))
        except:
            print(f"No elemments with xpath found")
        time.sleep(2)

        for i in range(len(elements)):
            try:
                text = elements[i].text
                if text != '':
                    texts.append(text)
                else:
                    texts.append(np.nan)
            except NoSuchElementException:
                text = np.nan
            try:
                headline = elements[i].find_element(
                    By.XPATH, ".//div[@class='kb0PBd A9Y9g']").text
            except NoSuchElementException:
                headline = np.nan
            headlines.append(headline)
            try:
                link = elements[i].find_element(
                    By.XPATH, ".//a[@jsname='UWckNb']").get_attribute("href")
            except NoSuchElementException:
                link = np.nan
            links.append(link)
        k = s + 1
        count_elements = count_elements + len(elements)
        print(f"page {k} done with {len(elements)
                                    } elements found. Total elements: {count_elements}")
        try:
            driver.find_element(By.XPATH, '//*[@id="pnnext"]').click()
            time.sleep(1)
        except:
            print("end of pages")
            break

    texts = pd.Series(texts, name='texts')
    links = pd.Series(links, name='links')
    headlines = pd.Series(headlines, name='headlines')
    print('\033[92mlinks and texts successfully acquired... \033[0m\n')

    return texts, links, headlines


def date_transform(text):
    if type(text) == float:
        return np.nan
    else:
        date = text.split('—')[0].replace(',', '')
        if "minutes ago" in date or 'hours ago' in date:
            return str(datetime.now().date()).split(' ')[0]
        elif "days ago" in date:
            days_ago = int(text.split()[0])
            return str(datetime.now() - timedelta(days=days_ago)).split(' ')[0]
        elif 'day ago' in date:
            return str(datetime.now().date()).split(' ')[0]
        else:
            # , format='%b %d %Y', errors='coerce')
            return str(pd.to_datetime(date)).split(' ')[0]
