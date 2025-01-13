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
import dateparser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


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
        driver.get("https://www.google.com/search?&tbm=nws")
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


def get_text_and_links2(driver, pg_num=3):
    links = []
    texts = []
    headlines = []
    dates = []
    k = 0
    count_elements = 0

    for s in range(pg_num):
        try:
            elements = WebDriverWait(driver, 4).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='SoaBEf']")))
        except:
            print(f"No elemments with xpath found")
            elements = ''
            pass
        time.sleep(2)
        for i in range(len(elements)):
            if elements != '':
                try:
                    text = elements[i].find_element(
                        By.XPATH, ".//div[@class='GI74Re nDgy9d']").text
                    if text != '':
                        texts.append(text)
                    else:
                        texts.append(np.nan)
                except:
                    texts.append(np.nan)
                    pass

                try:
                    headline = elements[i].find_element(
                        By.XPATH, ".//div[@class='n0jPhd ynAwRc MBeuO nDgy9d']").text
                    if headline != '':
                        headlines.append(headline)
                    else:
                        headlines.append(np.nan)
                except:
                    headlines.append(headline)
                    pass

                try:
                    link = elements[i].find_element(
                        By.XPATH, ".//a[@jsname='YKoRaf']").get_attribute("href")
                    if link != '':
                        links.append(link)
                    else:
                        links.append(np.nan)
                except:
                    links.append(np.nan)
                    pass

                try:
                    date = elements[i].find_element(
                        By.XPATH, ".//div[@class='OSrXXb rbYSKb LfVVr']").text
                    if date != '':
                        dates.append(date)
                    else:
                        dates.append(np.nan)
                except:
                    dates.append(np.nan)
                    pass
            else:
                pass

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
    dates = pd.Series(dates, name='dates')

    print('\033[92mlinks and texts successfully acquired... \033[0m\n')

    return texts, links, headlines, dates


def date_transform(text):
    if type(text) == float:
        return np.nan
    else:
        date = text.split('—')[0].replace(',', '')
        if "minutes ago" in date or 'hours ago' in date:
            return str(datetime.now().date()).split(' ')[0]
        elif 'hour ago' in date:
            return str(datetime.now().date()).split(' ')[0]
        elif "days ago" in date:
            days_ago = int(text.split()[0])
            return str(datetime.now() - timedelta(days=days_ago)).split(' ')[0]
        elif 'day ago' in date:
            return str(datetime.now().date()).split(' ')[0]
        else:
            # , format='%b %d %Y', errors='coerce')
            return str(pd.to_datetime(date)).split(' ')[0]


def converter_data(data):
    if "há" in data or "atrás" in data:
        # Usa dateparser para datas relativas
        return dateparser.parse(data)
    else:
        # Usa datetime.strptime para datas no formato "dd de mmm. de yyyy"
        return datetime.strptime(data, "%d de %b. de %Y")


def metrics_plot(y_test, y_pred, y_train, y_pred_train, pipeline):
    RMSE = (sum((y_pred - y_test)**2)/len(y_test)**0.5)
    RMSE_train = (sum((y_pred_train - y_train)**2)/len(y_train)**0.5)
    MAE = mean_absolute_error(y_test, y_pred)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_pred_train)

    # Exibindo os resultados
    print(f"RMSE (test): {RMSE:.3f}")
    print(f"RMSE (train): {RMSE_train:.3f}")
    print(f"MAE (test): {MAE:.3f}")
    print(f"MAE (train): {MAE_train:.3f}")
    print(f"R² (test): {r2_test:.3f}")
    print(f"R² (train): {r2_train:.3f}")
    print('\nmetrics successfully calculated...\n')
    # - Visualizações para análise de erros
    # Resíduos
    residuals = y_test - y_pred
    residuals_train = y_train - y_pred_train

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 2, 1)
    sns.histplot(y_test, kde=True, color='blue', bins=30)
    plt.title('Distribuição dos valores - Test')
    plt.xlabel('valores')
    plt.ylabel('Frequência')

    plt.subplot(3, 2, 2)
    sns.histplot(y_train, kde=True, color='red', bins=30)
    plt.title('Distribuição dos valores - Train')
    plt.xlabel('valores')
    plt.ylabel('')

    plt.subplot(3, 2, 3)
    sns.histplot(residuals, kde=True, color='blue', bins=30)
    plt.title('Distribuição dos Resíduos - Test')
    plt.xlabel('Resíduo')
    plt.ylabel('Frequência')

    plt.subplot(3, 2, 4)
    sns.histplot(residuals_train, kde=True, color='red', bins=30)
    plt.title('Distribuição dos Resíduos - Train')
    plt.xlabel('Resíduo')
    plt.ylabel('')

    plt.subplot(3, 2, 5)
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Previsões vs Real - Teste')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')

    # Treinamento
    plt.subplot(3, 2, 6)
    plt.scatter(y_train, y_pred_train, color='red', alpha=0.7)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
    plt.title('Previsões vs Real - Treinamento')
    plt.xlabel('Valores Reais')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

    # - Análise de importância das variáveis (feature importance)
    # Exibindo a importância das variáveis do modelo Random Forest
    importances = pipeline.named_steps['clf'].feature_importances_
    features = pipeline.named_steps['clf'].feature_names_in_

    # Organizando as variáveis por importância
    indices = importances.argsort()

    # Plotando
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(features)), importances[indices], align='center')
    plt.yticks(range(len(features)), [features[i] for i in indices])
    plt.title('Importância das Variáveis')
    plt.xlabel('Importância')
    plt.ylabel('Variáveis')
    plt.tight_layout()
    plt.show()

    print('--- End of program ---')
