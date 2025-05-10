# ML Multi-Factor Systematic Trading Strategy

## Project Overview

This project implements a multi-factor systematic trading strategy focusing on the Nasdaq 100 (QQQ). It leverages recurrent neural networks (LSTM), and sentiment scoress derived from news data using Large Language Models (LLMs) to predict market movements. The workflow involves several stages of data collection, processing, sentiment generation, and model training.

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
│   │   ├── nasdaq_news_data_processed.csv
│   │   ├── nasdaq_news_data_cleaned.csv (not uploaded due to size)
│   │   └── nasdaq_news_data_raw.csv (not uploaded due to size)
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
├── lstm_without_sentiment.ipynb
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
        *   This script filters data for the desired period (e.g., 2023), normalizes the market cap values (e.g., to millions of USD), and saves it to a CSV file. This output file contains the daily market capitalization ($M_{i,t}$) for each stock ($i$) on each day ($t$).
        *   This processed market capitalization is then used in downstream steps (e.g., in `src/generate_sentiment/sentiment_data_post.ipynb`) to calculate the market cap weight ($w_{i,t}$) for each stock $i$ on day $t$. The formula is:
            $$ w_{i,t} = \frac{M_{i,t}}{\sum_{j \in \text{Universe}_t} M_{j,t}} $$
            where:
            *   $M_{i,t}$ is the market capitalization of stock $i$ on day $t$.
            *   $\sum_{j \in \text{Universe}_t} M_{j,t}$ is the total market capitalization of all stocks $j$ belonging to the defined universe (Nasdaq 100 components) on day $t$.
*   **Output**: `data/market_cap/market_cap_2023.csv` (or a similar file representing the processed market capitalization data).

### 3. Sentiment Data Generation

*   **Objective**: Generate daily sentiment scores for Nasdaq 100 constituents based on news articles.
*   **Steps**:
    1.  `src/process_data/process_raw_data.py`: Initial processing of raw news data. This involves cleaning, filtering, and structuring the news articles.
        *   Input: Raw news data (from `data/news/`).
    2.  `src/process_data/process_cleaned_data.py`: Further processing of the cleaned news data, preparing it for sentiment analysis.
    3.  `src/EDA/news_eda.ipynb`: Exploratory Data Analysis on the processed news data to understand its characteristics.
    4.  `src/generate_sentiment/get_sentiment.py` (utilizing `src/LLM/llm.py`): Generates sentiment scores for news articles. The `llm.py` script handles interactions with a Large Language Model for sentiment classification.
        *   Input: Processed news data.
        *   Output: Sentiment scores for individual news items.
    5.  `src/EDA/sentiment_eda.ipynb`: Exploratory Data Analysis on the generated sentiment data.
    6.  `src/generate_sentiment/sentiment_data_post.ipynb`: Post-processes the sentiment scores, aggregating them daily and mapping them to the market cap data.
        *   This notebook calculates the final daily Nasdaq 100 weighted sentiment. The process is as follows:
            *   **Numerical Sentiment Mapping**: Raw sentiment labels (e.g., 'Strongly Bearish' to 'Strongly Bullish') are converted to numerical scores. Let $S'_{i,t,k}$ be this score for the $k$-th news item of stock $i$ on day $t$.
            *   **Daily Average Sentiment per Stock**: For each stock $i$, the numerical sentiment scores from all its news items on day $t$ are averaged:
                $$ \text{AvgSent}_{i,t} = \frac{1}{N_{i,t}} \sum_{k=1}^{N_{i,t}} S'_{i,t,k} $$
                where $N_{i,t}$ is the number of news items for stock $i$ on day $t$.
            *   **Market Capitalization Weighting**: Each stock's average daily sentiment ($\text{AvgSent}_{i,t}$) is then weighted by its market capitalization weight ($w_{i,t}$) for that day. The market cap weight $w_{i,t}$ represents stock $i$'s market cap as a proportion of the total market cap of the considered universe on day $t$.
            *   **Daily Nasdaq 100 Weighted Sentiment**: The final daily weighted sentiment for the Nasdaq 100 ($S_{\text{Nasdaq100},t}$) is the sum of these weighted sentiments for all stocks $i$ included in the analysis for day $t$:
                $$ S_{\text{Nasdaq100},t} = \sum_{i} (\text{AvgSent}_{i,t} \times w_{i,t}) $$
*   **Output**: `data/sentiment/nasdaq_100_daily_sentiment.csv`. This file contains daily aggregated sentiment scores for Nasdaq 100 stocks.

### 4. Main Prediction Pipeline

*   **Objective**: Train and evaluate LSTM models to predict QQQ price movements using the prepared features.
*   **Steps**:
    1.  `lstm_milestone2.ipynb`: This notebook represents an earlier milestone in developing the LSTM model. It involves feature engineering, model training, and evaluation using QQQ price data.
    2.  `lstm_with_sentiment.ipynb`: This notebook serves as the main pipeline. It integrates the QQQ price data and the `nasdaq_100_daily_sentiment.csv` to train and test the final LSTM model. This notebook would handle:
        *   Loading and merging QQQ price data and aggregated sentiment data.
        *   Feature scaling and preparation for the LSTM model.
        *   LSTM model, hyperparameter tuning, and training.
        *   Prediction and evaluation of model performance.
    3.  `lstm_without_sentiment.ipynb`: This notebook does not use sentiment data and only uses QQQ price data to train and test the final LSTM model. This notebook would handle:
        *   Loading QQQ price data.
        *   Feature scaling and preparation for the LSTM model.
        *   LSTM model, hyperparameter tuning, and training.
        *   Prediction and evaluation of model performance.
*   **Key Inputs**:
    *   QQQ historical price data. (from `data/data_for_LSTM/QQQ.csv`)
    *   Nasdaq 100 daily sentiment data. (from `data/data_for_LSTM/nasdaq_100_daily_sentiment.csv`)

## Data

The raw data and intermediate processed data are stored primarily within the `data/` directory:
*   `data/data_for_LSTM/`: Contains QQQ price data and Nasdaq 100 daily sentiment data. This is the final data used for LSTM models.
*   `data/market_cap/`: Contains market capitalization data, including `market_cap_2023.csv` and it raw data `tickers_2014_2025.csv`.
*   `data/news/`: Stores raw and processed news articles. Only the processed data is uploaded via github lfs. The raw data can be accessed from the original dataset [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset).
*   `data/sentiment/`: Contains generated sentiment scores by different LLMs and some intermediate files.
*   `data/nasdaq_100/`: Nasdaq 100 constituents stored in json file. 

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
    *   If `src/LLM/llm.py` uses API keys (e.g., for Gemini), add them to the `.env` file. Example:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

4.  **Data Acquisition:**
    *   Ensure all necessary raw data (market cap, news, QQQ prices) is downloaded and placed in the appropriate subdirectories within `data/`. Refer to the specific scripts or a data acquisition guide if available.

## How to Run
To run the LSTM model with sentiment, You can directly jump to the step 4 below. (All the final data is already processed and stored in `data/data_for_LSTM/`)
You can optionally run the steps 1-4 to preprocess more/different data before running the main pipeline.
1.  **Data Processing (optional):**
    *   Execute the scripts in the `src/process_data/` directory in the specified order as described in the "Workflow" section.
2.  **Sentiment Generation (optional):**
    *   Run `src/generate_sentiment/get_sentiment.py`.
    *   Execute the `src/generate_sentiment/sentiment_data_post.ipynb` notebook to finalize sentiment data.
3.  **Exploratory Data Analysis (Optional):**
    *   Run the EDA notebooks in `src/EDA/` (`news_eda.ipynb`, `sentiment_eda.ipynb`).
4.  **Main Pipeline:**
    *   Run the `lstm_with_sentiment.ipynb` notebook for the LSTM model with sentiment training and evaluation.
    *   Run the `lstm_without_sentiment.ipynb` notebook for the LSTM model without sentiment training and evaluation.
## Team Members
- Alan Chen zc2610@nyu.edu
- Jun Kwon jk7351@nyu.edu
- Vedant Desai vd2152@nyu.edu

## Citation
```bibtex
@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series}, 
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}
```