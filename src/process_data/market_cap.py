# %%
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(''))

market_cap_file_path = '../../data/market_cap/tickers_2014_2025.csv'
market_cap_df = pd.read_csv(market_cap_file_path)
print('Successfully loaded market cap data')
# %%
def get_market_cap_by_year(year: int) -> pd.DataFrame:
    """
    Get market cap data for a specific year
    """
    market_cap_df['date'] = pd.to_datetime(market_cap_df['date'])
    market_cap = market_cap_df[market_cap_df['date'].dt.year == year]
    market_cap = market_cap[['date', 'ticker', 'market_cap']]
    return market_cap

# %%
market_cap_2023 = get_market_cap_by_year(2023)
# %%
def normalize_market_cap(market_cap: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize market cap data
    """
    market_cap['market_cap'] = market_cap['market_cap'] / 100000000
    market_cap_normalized = market_cap.rename(columns={'market_cap': 'market_cap(millions)', 'date': 'Date', 'ticker': 'Ticker'})
    return market_cap_normalized

# %%
market_cap_normalized = normalize_market_cap(market_cap_2023)
# %%
market_cap_normalized.to_csv('../../data/market_cap/market_cap_2023.csv', index=False)

# The 'market_cap(millions)' column in the output CSV ('market_cap_2023.csv')
# is used to calculate the market capitalization weight for each stock on each day.
# This weight is typically used in downstream analyses, such as calculating
# a market-cap-weighted sentiment score for an index.
#
# The formula to calculate the market cap weight ($w_{i,t}$) for stock $i$ on day $t$ is:
#
# $$ w_{i,t} = \frac{M_{i,t}}{\sum_{j \in \text{Universe}_t} M_{j,t}} $$
#
# where:
# - $M_{i,t}$ is the market capitalization of stock $i$ on day $t$
#   (represented by the 'market_cap(millions)' column from this script's output).
# - $\sum_{j \in \text{Universe}_t} M_{j,t}$ is the total market capitalization
#   of all stocks $j$ belonging to the defined universe (e.g., Nasdaq 100 components)
#   on day $t$.
#
# This script prepares the $M_{i,t}$ values. The actual calculation of $w_{i,t}$
# and its addition as a new column (e.g., 'market_cap_weight') typically occurs
# in a downstream processing step (e.g., in the sentiment processing notebook).
# %%
