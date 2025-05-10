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
