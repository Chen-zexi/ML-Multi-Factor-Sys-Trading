# %%
import pandas as pd
import json
# %%
input_file_path = 'data/news/nasdaq_news_data_raw.csv'
output_file_path = 'data/news/nasdaq_news_data_cleaned.csv'
json_file_path = 'data/nasdaq_100/nasdaq_100_members_2015_2024.json'
chunk_size = 10000
filtered_data = []

start_date = pd.Timestamp('2015-01-01', tz='UTC')
end_date = pd.Timestamp('2024-01-01', tz='UTC')

print(f"Reading data from {input_file_path} in chunks of {chunk_size}...")
print(f"Filtering for dates between {start_date.date()} (inclusive) and {end_date.date()} (exclusive, UTC)...")

try:
    chunk_iter = pd.read_csv(input_file_path, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i+1}...")

        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)

        initial_rows = len(chunk)
        chunk.dropna(subset=['Date', 'Stock_symbol', 'Article'], inplace=True)
        rows_dropped = initial_rows - len(chunk)
        if rows_dropped > 0:
            print(f"  Dropped {rows_dropped} rows with NaN in 'Date', 'Stock_symbol', or 'Article'.")

        filtered_chunk = chunk[(chunk['Date'] >= start_date) & (chunk['Date'] < end_date)]

        if not filtered_chunk.empty:
            filtered_data.append(filtered_chunk)


    raw_df = pd.concat(filtered_data, ignore_index=True)
         
    raw_df = raw_df.sort_values('Date')
    raw_df = raw_df.rename(columns={'Stock_symbol': 'Ticker'})
    raw_df.drop(columns=['Url'], inplace=True, errors='ignore')
    raw_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    raw_df.drop(columns=['Publisher'], inplace=True, errors='ignore')
    raw_df.drop(columns=['Author'], inplace=True, errors='ignore')
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.strftime('%Y-%m-%d')
    print(f"Performed basic cleaning on all data")
    
except FileNotFoundError:
    print(f"Error: File not found at {input_file_path}")
    df_filtered = pd.DataFrame()
except Exception as e:
    print(f"An error occurred: {e}")
    df_filtered = pd.DataFrame()
    
    
# %% [markdown]
# Match with Nasdaq 100 members


    
# %%
# 1. Load the Nasdaq membership data from JSON

try:
    with open(json_file_path, 'r') as f:
        nasdaq_membership = json.load(f)
    print(f"Successfully loaded membership data from {json_file_path}")

    # Convert keys (years) to int and values to sets
    nasdaq_membership_processed = {}
    for year_str, tickers in nasdaq_membership.items():
        if year_str.isdigit() and year_str != '2024':
            year_int = int(year_str)
            nasdaq_membership_processed[year_int] = set(tickers)
        else:
            print(f"Warning: Skipping non-integer key '{year_str}' in JSON.")

    print(f"Membership years considered for filtering: {sorted(nasdaq_membership_processed.keys())}")

except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file_path}")
    nasdaq_membership_processed = {}
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_file_path}")
    nasdaq_membership_processed = {}
except Exception as e:
    print(f"An unexpected error occurred loading the JSON: {e}")
    nasdaq_membership_processed = {}
    

# %%
# 2. Filter the news data based on Nasdaq membership
news_df = raw_df.copy()

filtered_news_df = pd.DataFrame()

if not news_df.empty and nasdaq_membership_processed:
    # Make a copy to avoid modifying the original df if it's used elsewhere
    df_to_filter = news_df.copy()

    # Convert 'Date' column to datetime objects
    df_to_filter['Date'] = pd.to_datetime(df_to_filter['Date'], errors='coerce')

    # Drop rows where date conversion failed
    original_rows = len(df_to_filter)
    df_to_filter = df_to_filter.dropna(subset=['Date'])
    if len(df_to_filter) < original_rows:
        print(f"Warning: Dropped {original_rows - len(df_to_filter)} rows with unparseable dates.")

    # Extract the year
    df_to_filter['Year'] = df_to_filter['Date'].dt.year

    # Define the filtering function
    def is_ticker_valid_for_year(row):
        year = row['Year']
        ticker = row['Ticker']

        if year in nasdaq_membership_processed:
            return ticker in nasdaq_membership_processed[year]
        return False # Year not in our membership keys

    # Apply the filter
    print("\nApplying filter based on yearly Nasdaq membership")
    # Use boolean indexing directly which can be faster than apply for this case
    # Create a boolean mask
    mask = df_to_filter.apply(is_ticker_valid_for_year, axis=1)
    filtered_news_df = df_to_filter[mask].copy()

    # Keep the 'Year' column for the next step (finding missing tickers)
    # If you want to drop it later: filtered_news_df = filtered_news_df.drop(columns=['Year'])

    print("\nFiltered DataFrame Info:")
    filtered_news_df.info()


elif news_df.empty:
     print("\nNews DataFrame is empty. Skipping filtering.")
else: # nasdaq_membership_processed is empty or only had 2025
     print("\nNasdaq membership data  could not be loaded or is empty. Skipping filtering.")

# %%
filtered_news_df.to_csv(output_file_path, index=False)
print(f"Filtered data saved to '{output_file_path}'")


# %%
# 3. Find missing tickers per year
missing_tickers_by_year = {}
news_tickers_by_year = {} # Store unique tickers found in news data per year

if not filtered_news_df.empty and 'nasdaq_membership_processed' in locals() and nasdaq_membership_processed:
    print("\nComparing news data tickers against membership list for each year...")

    # --- Start Replace Block ---
    # Check if the required 'Year' column exists and is integer
    if 'Year' in filtered_news_df.columns and pd.api.types.is_integer_dtype(filtered_news_df['Year']):
        year_col_to_use = 'Year' # Use the existing integer 'Year' column directly
        print(f"Using integer column '{year_col_to_use}' for comparison with integer membership keys.")
    elif 'Year' in filtered_news_df.columns:
         # If it exists but is not integer, flag potentially unexpected behavior
         print(f"Warning: Using existing column 'Year' which is not integer type ({filtered_news_df['Year'].dtype}). Comparison might fail as membership keys are integers.")
         year_col_to_use = 'Year' # Still attempt to use it, but warn
    else:
        print("Error: 'Year' column not found in filtered_news_df. Cannot perform comparison.")
        year_col_to_use = None # Prevent further processing
    # --- End Replace Block ---

    if year_col_to_use:
        # Iterate through the years we have membership data for (integer keys)
        years_to_check = sorted(nasdaq_membership_processed.keys()) # These are integers

        print("\nUnique Tickers found in News Data per Year:")
        for year in years_to_check: # 'year' is an integer
            # Get the set of official members for the year (integer key)
            official_members = nasdaq_membership_processed.get(year, set())
            if not isinstance(official_members, set):
                 official_members = set(official_members) # Ensure it's a set

            # Get the set of unique tickers present in the filtered news data for that integer year
            # Compare integer 'year' with integer 'Year' column ('year_col_to_use')
            news_tickers_in_year_series = filtered_news_df[filtered_news_df[year_col_to_use] == year]['Ticker']
            news_tickers_in_year = set(news_tickers_in_year_series.unique())

            # Store and print the count for the year
            news_tickers_by_year[year] = sorted(list(news_tickers_in_year))
            print(f"  - Year {year}: Found {len(news_tickers_in_year)} unique tickers in news data.")

            # Find tickers in the official list but not in the news data for that year
            missing_tickers = official_members - news_tickers_in_year

            if missing_tickers:
                missing_tickers_by_year[year] = sorted(list(missing_tickers))
                example_missing = list(missing_tickers)[:5]
                print(f"    - Missing {len(missing_tickers)} official members (e.g., {example_missing}{'...' if len(missing_tickers) > 5 else ''}).")
            else:
                print(f"    - All {len(official_members)} official members found in news data.")
                missing_tickers_by_year[year] = []

        # Save the unique news tickers found per year to JSON (using integer keys)
        news_tickers_output_path = 'data/nasdaq_100/available_news_tickers_by_year.json'
        try:
            # Convert integer keys to strings for JSON compatibility if needed
            news_tickers_by_year_json = {str(k): v for k, v in news_tickers_by_year.items()}
            with open(news_tickers_output_path, 'w') as f:
                json.dump(news_tickers_by_year_json, f, indent=4)
            print(f"\nSaved unique news tickers per year to '{news_tickers_output_path}'")
        except Exception as e:
            print(f"\nError saving news tickers JSON: {e}")


        # Display the full list of missing tickers per year
        print("\nDetailed Missing Official Tickers by Year:")
        if missing_tickers_by_year:
            all_missing_count = 0
            # Convert integer keys to strings for display if desired, or keep as int
            for year, tickers in sorted(missing_tickers_by_year.items()):
                if tickers:
                     print(f" - {year} ({len(tickers)} missing): {tickers}")
                     all_missing_count += len(tickers)
                else:
                     print(f" - {year}: None")
            print(f"Total missing official tickers across checked years: {all_missing_count}")
        else:
            print("No years were checked or no missing tickers found.")

elif filtered_news_df.empty:
    print("\nFiltered news DataFrame is empty. Cannot perform missing ticker comparison.")
else:
    print("\nMembership data ('nasdaq_membership_processed') not available or empty. Cannot perform missing ticker comparison.")




