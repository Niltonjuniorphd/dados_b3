### README.md


# Dados B3

This repository contains scripts and data analysis tools for working with financial data from B3.

## Instructions

### 1. Clone the Repository
To clone this repository to your local machine, use the following command:

```bash
git clone https://github.com/Niltonjuniorphd/dados_b3.git
cd dados_b3
```

### 2. Create a Virtual Environment
Create a virtual environment using `venv` to manage dependencies:
```bash
python -m venv .env
```

Activate the virtual environment:
(not required if using VSCode in Windows)
- On Linux/Mac:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Required Libraries
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Run the Script
Execute the main script `run.py` to start the application:
```bash
python run.py
```

## Notes
- Ensure you have Python 3.8 or higher installed on your system before proceeding.
- The script will create a CSV files `root` directory.
- The external file from the B3 website must be stored in the `root` directory (see next section).
---

## How it Works

### **Step 1: Scrap and Persist Data**
The script **`search_news.py`** performs the following
1. The script **`search_news.py`** fetches news from **Google News** using Selenium. The **`focus_key`** variable defines the search keywords.
2. Once the site responds, the script scrapes news articles, collecting **headlines**, **links**, and **text**. It processes at least **30 pages** and generates a CSV file named **`news_df_aaa_mm_dd.csv`** to store the data.

---

### **Step 2: Create Features and Persist as CSV**
The script **`create_features.py`** performs the following:
1. **Sentiment Analysis**: Reads data from **`news_df_aaa-mm-dd.csv`** and calculates sentiment scores using **TextBlob** and **VADER** tools. Articles are classified as **positive**, **negative**, or **neutral**.
2. **Metric Calculation**: Computes word counts, word frequency, and other metrics.
3. **Financial Data Integration**: Loads financial data from **`2024_ExchangeRateFile_20241227_1.csv`** (downloaded from [B3's official site](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/boletim-diario/historico-de-taxas-de-cambio-resolucao-bcb-n-120/)) and merges it with news sentiment data based on the **date** column.
4. **Output**: Creates a merged DataFrame with features such as **daily returns** and saves it as **`news_df_features_aaa-mm-dd.csv`**.

---

### **Step 3: Train and Test the Model**
The script **`train_model.py`** performs the following:
1. **Data Preparation**: Reads data from **`news_df_features_aaa-mm-dd.csv`** and splits it into **training** and **testing** sets.
2. **Model Training**: Uses **GridSearchCV** to optimize hyperparameters for a **Random Forest Classifier**. The model is trained on the training set.
3. **Evaluation**: Calculates and prints the **accuracy** on the testing set. Statistics and **plots** are displayed in sequential windows. (Close one plot window to view the next.)

--- 

### Executing the Script **'run.py'** will perform the steps above.



