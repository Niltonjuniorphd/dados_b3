#%%
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from functions import (call_driver, send_focus_key,
                       get_text_and_links, date_transform, get_text_and_links2)

driver = call_driver()

focus_key = 'Lula economia dolar'

send_focus_key(driver, focus_key=focus_key)

elements = WebDriverWait(driver, 4).until(
    EC.presence_of_all_elements_located((By.XPATH, "//div[@class='MjjYud']"))
)

#%%
# Inspecionando cada elemento em elements
for index, element in enumerate(elements):
    print(f"\n=== Elemento {index + 1} ===")
    
    # Imprime o texto do elemento
    print("Texto:", element.text)
    
    # Imprime os atributos disponíveis (exemplo: href, class)
    try:
        print("Link (href):", element.find_elements(
                    'xpath', "//a[@jsname='UWckNb']"
                )[index].get_attribute("href"))
    except:
        print("Este elemento não possui atributo 'href'")
    
    print("Classe:", element.get_attribute("class"))

# %%
# Obtém o HTML completo do elemento
from bs4 import BeautifulSoup

# Itera sobre os elementos e formata o HTML
for index, element in enumerate(elements):
    raw_html = element.get_attribute("outerHTML")
    formatted_html = BeautifulSoup(raw_html, 'html.parser').prettify()  # Formata o HTML
    
    print(f"\n=== Elemento {index + 1} ===")
    print(formatted_html)


# %%

elements = WebDriverWait(driver, 4).until(EC.presence_of_all_elements_located((By.XPATH, "//div[@class='MjjYud']")))

print(elements[2].find_element('xpath', "//a[@jsname='UWckNb']").get_attribute("href"))


# %%
print(elements[0].text)

links = get_text_and_links3(driver, pg_num=1)
# %%
