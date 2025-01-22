# Financial News Analysis and Stock Prediction

![Financial Insights](images/stock1.jpg "Financial News Analysis")



**A Project Exploring the Relationship Between Financial News and Stock Market Movements**

This project analyzes financial news to forecast stock price movements by leveraging natural language processing (NLP) techniques and machine learning models. The analysis explores the relationship between news sentiment and stock market performance, providing actionable insights for investors and analysts.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Sentiment Analysis](#sentiment-analysis)
    - [Feature Engineering](#feature-engineering)
    - [Model Training and Evaluation](#model-training-and-evaluation)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Stock Ticker Extraction Methodology](#stock-ticker-extraction-methodology)
8. [Financial Insights and Forecasting](#financial-insights-and-forecasting)
9. [Correlation of News with Stock Performance](#correlation-of-news-with-stock-performance)
10. [Conclusion and Recommendations](#conclusion-and-recommendations)
11. [How to Run the Notebook](#how-to-run-the-notebook)

---

## 1. Introduction

This project aims to:
- Analyze financial news articles using NLP techniques.
- Extract stock ticker information from the news.
- Perform sentiment analysis and time series forecasting to predict stock movements.
- Understand the correlation between news sentiment and stock performance.

---

## 2. Project Structure

The project consists of the following key sections:
1. **Exploratory Data Analysis (EDA):** Data inspection, cleaning, and visualization.
2. **Stock Ticker Extraction:** Identification and mapping of stock tickers using NLP.
3. **Sentiment Analysis:** Polarity analysis of news articles to measure sentiment.
4. **Forecasting Models:** Machine learning models to predict stock price movements.
5. **Correlation Analysis:** Quantifying the impact of news sentiment on stock performance.

---

## 3. Requirements

To replicate this analysis, you will need the following:
- Python (3.7+)
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk
  - WordCloud
  - sklearn
  - keras/tensorflow (if using deep learning models)

## 4. Dataset

This project utilizes three distinct datasets:

1. **Financial News Dataset:** 
    - **Source:** 
    - **Description:** This dataset contains financial news articles with the following attributes:
        - **Headlines:** Titles of financial news articles.
        - **Date:** Publication date of the news article.
        - **Description:** A detailed text description of the news article.

2. **Finance Online Dataset:**
    - **Source:** Downloaded within the project using `!pip install yfinance`
    - **Description:** This dataset is sourced from the `yfinance` library, providing historical stock price data for various companies.

3. **Top 50 American Stock Companies Dataset:**
    - **Source:** 
    - **Description:** This CSV file contains a list of the top 50 American stock companies with two columns:
        - **Companies:** Names of the top 50 companies.
        - **Tickers:** Stock tickers corresponding to each company. 

This combination of datasets allows for a comprehensive analysis of the relationship between financial news sentiment and stock price movements.
## 5. Methodology

### Data Preprocessing

1. **Data Loading:** Load the financial news data and stock price data into appropriate data structures (e.g., Pandas DataFrames).
2. **Data Cleaning:** 
    - Handle missing values (e.g., imputation, removal).
    - Remove duplicates and irrelevant data.
    - Clean the text data by removing special characters, HTML tags, and irrelevant symbols.
3. **Text Preprocessing:** 
    - Convert text to lowercase.
    - Tokenize the text into individual words.
    - Remove stop words (common words like "the," "a," "is").
    - Perform stemming or lemmatization to reduce words to their base forms.
4. **Feature Engineering:**
    - Create new features from the stock price data (e.g., moving averages, price differences).
    - Calculate technical indicators (e.g., RSI, MACD).

### Sentiment Analysis

1. **Sentiment Model Training:** 
    - Train a machine learning model (e.g., Naive Bayes, Support Vector Machine, LSTM) to classify news articles as positive, negative, or neutral based on their sentiment.
    - Utilize techniques like TF-IDF or word embeddings (e.g., Word2Vec, GloVe) for text representation.
2. **Sentiment Prediction:** 
    - Apply the trained sentiment model to the news articles in the dataset to predict their sentiment scores.

### Feature Engineering

1. **Combine Features:** Combine the predicted sentiment scores with the stock price features and other relevant features (e.g., trading volume, market indices).

### Model Training and Evaluation

1. **Split Data:** Split the data into training and testing sets.
2. **Model Selection:** Choose an appropriate machine learning model for stock price prediction (e.g., Linear Regression, Random Forest, LSTM).
3. **Model Training:** Train the chosen model on the training data.
4. **Model Evaluation:** Evaluate the model's performance on the testing data using appropriate metrics (e.g., Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared).


## Exploratory Data Analysis (EDA)
**Exploratory Data Analysis (EDA)** is a crucial step to better understand the dataset and prepare it for further analysis. In this project, we perform the following tasks during EDA:

1. **Data Inspection:**
   - Load the dataset and inspect its structure using basic pandas functions:
     - `df.info()` provides the dataset's information, including the number of rows, columns, and data types.
     - `df.head()` shows the first 5 rows of the dataset to get a preview of the data.
     - `df.columns` lists all column names.
   
2. **Handling Missing Values:**
   - If the **Date** or **Headline** of any article is missing, remove those articles.
   - If the **Description** is missing, replace the missing description with an empty string.

3. **Removing Duplicates:**
   - Remove any duplicate entries in the dataset to avoid skewed analysis.

4. **Unique Articles:**
   - Calculate the number of **unique articles** based on the **Headlines** column to ensure that articles are not repeated.

5. **Trend Analysis:**
   - Visualize the number of articles published on each day in 2020 to understand trends using **matplotlib.pyplot**.

6. **Word Frequency Analysis:**
   - Analyze the most frequent words in the **Description** column.
   - Generate a **WordCloud** to visualize the most frequent words and understand the common topics in the news articles.



## Results

- Present the results of the sentiment analysis model (e.g., accuracy, precision, recall, F1-score).
- Present the results of the stock price prediction model (e.g., performance metrics on the test set).
- Discuss the findings and insights gained from the analysis.




This README file clearly explains each section of the notebook while adhering to standard README formatting practices.


Install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn tensorflow






