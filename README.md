# ML Multi-Factor Systematic Trading Strategy

## Project Overview

This project implements a multi-factor systematic trading strategy focusing on the Nasdaq 100 (QQQ). It leverages recurrent neural networks (LSTM), and sentiment analysis derived from news data using Large Language Models (LLMs) to predict market movements. The workflow involves several stages of data collection, processing, sentiment generation, and model training.

## Project Structure

```
ML-Multi-Factor-Sys-Trading/
├── data/
│   ├── data_for_LSTM/
│   │   ├── nasdaq_100_daily_sentiment.csv
│   │   └── QQQ.csv
│   ├── market_cap/
│   │   ├── market_cap_2023.csv
│   │   └── tickers_2014_2025.csv
│   ├── nasdaq_100/
│   │   ├── nasdaq_100_members_2015_2024.json
│   │   ├── news_tickers_by_year.json
│   │   └── unique_nasdaq_tickers.json
│   ├── news/
│   │   ├── nasdaq_news_data_cleaned.csv
│   │   ├── nasdaq_news_data_processed.csv
│   │   └── nasdaq_news_data_raw.csv
│   └── sentiment/
│       ├── nasdaq_100_daily_sentiment.csv
│       ├── news_sentiment_results.csv
│       ├── news_sentiment_results_gemini.csv
│       ├── news_sentiment_results_gemini_hb.csv
│       └── sentiment_2023.csv
├── src/
│   ├── EDA/
│   │   ├── news_eda.ipynb
│   │   └── sentiment_eda.ipynb
│   ├── generate_sentiment/
│   │   ├── Get_sentiment.ipynb
│   │   ├── get_sentiment.py
│   │   └── sentiment_data_post.ipynb
│   ├── LLM/
│   │   ├── __pycache__/
│   │   └── llm.py
│   ├── Nasdaq_100/
│   │   └── find_nasdaq_constitution.py
│   └── process_data/
│       ├── market_cap.py
│       ├── process_cleaned_data.py
│       └── process_raw_data.py
├── .env
├── lstm_milestone2.ipynb
├── lstm_with_sentiment.ipynb
├── requirements.txt
└── README.md
```

## Workflow

The project follows a structured workflow to process various data sources and feed them into the predictive models:

### 1. QQQ Price Data Acquisition

*   **Objective**: Obtain historical price data for QQQ.
*   **Steps**:
    1.  Data Download: QQQ price data is downloaded (e.g., from WRDS).
*   **Output**: QQQ price data file (in `data/data_for_LSTM/`).

### 2. Market Cap Data Acquisition and Processing

*   **Objective**: Determine the constituents of the Nasdaq 100 and their market capitalization.
*   **Steps**:
    1.  `src/Nasdaq_100/find_nasdaq_constitution.py`: Identifies the tickers to be included in the Nasdaq 100.
    2.  Data Download: Market capitalization data is downloaded (e.g., from WRDS - Wharton Research Data Services).
    3.  `src/process_data/market_cap.py`: Processes the downloaded market cap data.
*   **Output**: `data/market_cap/market_cap_2023.csv` (or a similar file representing the processed market capitalization data).

### 3. Sentiment Data Generation

*   **Objective**: Generate daily sentiment scores for Nasdaq 100 constituents based on news articles.
*   **Steps**:
    1.  `src/process_data/process_raw_data.py`: Initial processing of raw news data. This involves cleaning, filtering, and structuring the news articles.
        *   Input: Raw news data (e.g., from `data/news/`).
    2.  `src/process_data/process_cleaned_data.py`: Further processing of the cleaned news data, preparing it for sentiment analysis.
    3.  `src/EDA/news_eda.ipynb`: Exploratory Data Analysis on the processed news data to understand its characteristics.
    4.  `src/generate_sentiment/get_sentiment.py` (utilizing `src/LLM/llm.py`): Generates sentiment scores for news articles. The `llm.py` script handles interactions with a Large Language Model for sentiment classification.
        *   Input: Processed news data.
        *   Output: Sentiment scores for individual news items.
    5.  `src/EDA/sentiment_eda.ipynb`: Exploratory Data Analysis on the generated sentiment data.
    6.  `src/generate_sentiment/sentiment_data_post.ipynb`: Post-processes the sentiment scores, aggregating them daily and mapping them to the market cap data.
*   **Output**: `data/sentiment/nasdaq_100_daily_sentiment.csv`. This file contains daily aggregated sentiment scores for Nasdaq 100 stocks.

### 4. Main Prediction Pipeline

*   **Objective**: Train and evaluate LSTM models to predict QQQ price movements using the prepared features.
*   **Steps**:
    1.  `LSTM_Milestone2.ipynb`: This notebook likely represents an initial phase or a specific milestone in developing the LSTM model. It might involve feature engineering, model training, and evaluation using various data sources.
    2.  `qqq_lstm_pipeline.ipynb`: This notebook serves as the main pipeline. It integrates the QQQ price data and the `nasdaq_100_daily_sentiment.csv` to train and test the final LSTM model. This notebook would handle:
        *   Loading and merging QQQ price data and aggregated sentiment data.
        *   Feature scaling and preparation for the LSTM model.
        *   LSTM model definition, training, and prediction.
        *   Evaluation of model performance.
*   **Key Inputs**:
    *   QQQ historical price data.
    *   `data/sentiment/nasdaq_100_daily_sentiment.csv`

## Data

The project utilizes several key data sources and intermediate files stored primarily within the `data/` directory:

*   `data/market_cap/`: Contains market capitalization data, including `market_cap_2023.csv`.
*   `data/news/`: Stores raw and processed news articles.
*   `data/sentiment/`: Contains generated sentiment scores, notably `nasdaq_100_daily_sentiment.csv`.
*   `data/nasdaq_100/`: Likely stores QQQ price data and other Nasdaq 100 specific information.
*   `data/data_for_LSTM/`: May contain consolidated datasets prepared specifically for LSTM model training.
*   `data/backup/`: For storing backups of important data files.

## Setup and Installation

1.  **Clone the repository (if applicable).**

2.  **Set up your Python environment and install dependencies:**

    Choose one of the following methods:

    *   **Using `uv` (recommended for speed):**
        ```bash
        # Install uv if you haven't already (see https://github.com/astral-sh/uv)
        # Create and activate a virtual environment
        uv venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
        # Install dependencies
        uv pip install -r requirements.txt
        ```

    *   **Using `pip`:**
        ```bash
        # Create and activate a virtual environment (e.g., using venv)
        # python -m venv myenv
        # source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

        # Install dependencies
        pip install -r requirements.txt
        ```

3.  **Set up environment variables:**
    *   Create a `.env` file in the root directory.
    *   If `src/LLM/llm.py` uses API keys (e.g., for Google Generative AI), add them to the `.env` file. Example:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

4.  **Data Acquisition:**
    *   Ensure all necessary raw data (market cap, news, QQQ prices) is downloaded and placed in the appropriate subdirectories within `data/`. Refer to the specific scripts or a data acquisition guide if available.

## How to Run
To run the LSTM model with sentiment, You can directly jump to the step 5 below.
You can optionally run the steps 1-4 to preprocess more/different data before running the main pipeline.
1.  **Data Processing (optional):**
    *   Execute the scripts in the `src/process_data/` directory in the specified order (e.g., `market_cap.py`, `process_raw_data.py`, `process_cleaned_data.py`).
    *   Run `src/Nasdaq_100/find_nasdaq_constitution.py`.
2.  **Sentiment Generation (optional):**
    *   Run `src/generate_sentiment/get_sentiment.py`.
    *   Execute the `src/generate_sentiment/sentiment_data_post.ipynb` notebook to finalize sentiment data.
3.  **Exploratory Data Analysis (Optional):**
    *   Run the EDA notebooks in `src/EDA/` (`news_eda.ipynb`, `sentiment_eda.ipynb`).
4.  **Main Pipeline:**
    *   Run the `lstm_with_sentiment.ipynb` notebook for the main model training and evaluation.

## Team Members
- Alan Chen zc2610@nyu.edu
- Jun Kwon jk7351@nyu.edu
- Vedant Desai vd2152@nyu.edu