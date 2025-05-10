# %%
import pandas as pd
from typing import List
import time
from tqdm.notebook import tqdm
import math
import sys
import os
sys.path.append(os.path.abspath('../'))
from LLM.llm import get_sentiment, get_batch_sentiment

# %%
first_processed_uid = 0
last_processed_uid = 0
sentiment_file_path = '../../data/sentiment/sentiment_2023.csv'
processed_news_file_path = '../../data/news/nasdaq_news_data_processed.csv'
## Config further detail in the bottom of the file. 


existing_sentiment_df = pd.read_csv(sentiment_file_path)
first_processed_uid = existing_sentiment_df['UID'].min()
last_processed_uid = existing_sentiment_df['UID'].max()
print(f"Initialized first_processed_uid from existing file: {first_processed_uid}")
print(f"Initialized last_processed_uid from existing file: {last_processed_uid}")


print(f"Loading processed news data from {processed_news_file_path}...")
all_processed_news_df = pd.read_csv(processed_news_file_path, parse_dates=['Date'])
print(f"Loaded {len(all_processed_news_df)} news items.")


# %%
def parse_news_for_llm(df_chunk: pd.DataFrame):
    required_parse_cols = ['Ticker', 'Title', 'Summary']
    news_list = df_chunk[required_parse_cols].to_dict(orient='records')
    return news_list


def analyze_sentiment_wrapper(df_chunk: pd.DataFrame, mode: str, model_name="grok-3-beta") -> pd.DataFrame:
    original_data = df_chunk[['UID', 'Date', 'Ticker']].reset_index(drop=True)

    parsed_news = parse_news_for_llm(df_chunk)

    results: List[Sentiment] = []

    if mode == 'single':
        print(f"Processing {len(parsed_news)} items in single mode using {model_name}...")
        with tqdm(total=len(parsed_news), desc="Single Processing") as pbar:
            for news_item in parsed_news:
                sentiment_result = get_sentiment(news_item, model_name)
                results.append(sentiment_result)
                pbar.update(1)

    elif mode == 'batch':
        print(f"Processing {len(parsed_news)} items in batch mode using {model_name}...")
        batch_response = get_batch_sentiment(parsed_news, model_name)
        sentiment_results = batch_response.sentiments 

        if len(sentiment_results) != len(parsed_news):
             print(f"Warning: Batch result count ({len(sentiment_results)}) doesn't match input count ({len(parsed_news)}). Alignment might be incorrect.")
        results.extend(sentiment_results[:len(parsed_news)]) 


    output_data = [{'Sentiment': r.sentiment, 'Reason': r.reason} for r in results]
    sentiment_df = pd.DataFrame(output_data)

    if sentiment_df.empty or len(sentiment_df) != len(original_data):
         print(f"Warning: Sentiment analysis resulted in {len(sentiment_df)} items, expected {len(original_data)}. Returning original data with NA sentiment.")
         final_df = original_data.copy()
         final_df['Sentiment'] = pd.NA
         final_df['Reason'] = pd.NA
    else:
        final_df = original_data.join(sentiment_df)

    return final_df[['UID', 'Date', 'Ticker', 'Sentiment', 'Reason']]


# %%
# Add this helper function BEFORE the process_sentiment_by_uid function definition
def _save_current_progress(base_df_for_saving: pd.DataFrame, new_results_list: list, output_path: str, main_overwrite_flag: bool):
    """Helper to save intermediate sentiment results."""
    if not new_results_list:
        print("No new results generated in current list, nothing to save for progress.")
        return

    print("Attempting to save intermediate progress...")
    try:
        temp_new_results_df = pd.concat(new_results_list, ignore_index=True)
        if temp_new_results_df.empty:
            print("Aggregated new results are empty, nothing to save for progress.")
            return

        # New results should always take precedence for the UIDs they contain.
        uids_in_temp_new_results = temp_new_results_df['UID'].unique()
        
        # Filter base_df_for_saving to remove any UIDs that are now in temp_new_results_df
        base_df_for_saving_filtered = base_df_for_saving[~base_df_for_saving['UID'].isin(uids_in_temp_new_results)]
        
        temp_combined_df = pd.concat([base_df_for_saving_filtered, temp_new_results_df], ignore_index=True)

        # Safeguard: drop duplicates by UID, keeping the latest (from new results if any overlap)
        initial_count = len(temp_combined_df)
        temp_combined_df.drop_duplicates(subset='UID', keep='last', inplace=True)
        if len(temp_combined_df) < initial_count:
            print(f"(Intermediate save) Removed {initial_count - len(temp_combined_df)} duplicate UIDs by keeping 'last'.")

        temp_combined_df.sort_values(by='UID', inplace=True)
        if 'UID' in temp_combined_df.columns and not temp_combined_df.empty:
            temp_combined_df['UID'] = temp_combined_df['UID'].astype(int) # Assuming UIDs are non-nullable by this stage

        temp_combined_df.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        print(f"Intermediate progress saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving intermediate progress: {e}")


# %%
def process_sentiment_by_uid(
    processed_news_df: pd.DataFrame,
    batch_size: int,
    start_uid: int,
    end_uid: int,
    overwrite: bool = False,
    model_name: str = "gemini-2.0-flash",
    sentiment_output_path: str = 'data/sentiment/news_sentiment_results_gemini.csv'
) -> None:
    global last_processed_uid

    print(f"\n--- Starting Sentiment Processing ---")
    print(f"Range: UID {start_uid} to {end_uid}")
    print(f"Batch Size: {batch_size}")
    print(f"Overwrite: {overwrite}")
    print(f"Model: {model_name}")
    print(f"Output Path: {sentiment_output_path}")

    expected_cols = ['UID', 'Date', 'Ticker', 'Sentiment', 'Reason']
    dtype_spec = {'UID': 'Int64', 'Date': 'datetime64[ns]', 'Ticker': object, 'Sentiment': object, 'Reason': object}

    try:
        existing_sentiment_df = pd.read_csv(sentiment_output_path, parse_dates=['Date'])
        # Ensure 'Sentiment' and 'Reason' columns exist, adding them as object type if not
        for col in ['Sentiment', 'Reason']:
            if col not in existing_sentiment_df.columns:
                existing_sentiment_df[col] = pd.NA
        
        # Ensure correct dtypes for critical columns
        if 'UID' in existing_sentiment_df.columns:
            existing_sentiment_df['UID'] = existing_sentiment_df['UID'].astype('Int64')
        else: # Add UID column if somehow missing, though unlikely if file has data
            existing_sentiment_df['UID'] = pd.Series(dtype='Int64')

        for col in ['Sentiment', 'Reason']: # Ensure these are object for pd.NA compatibility
             if col in existing_sentiment_df.columns:
                existing_sentiment_df[col] = existing_sentiment_df[col].astype(object)

    except FileNotFoundError:
        print(f"Sentiment file {sentiment_output_path} not found. Starting with an empty sentiment dataframe.")
        existing_sentiment_df = pd.DataFrame(columns=expected_cols)
        for col, dtype_val in dtype_spec.items():
            existing_sentiment_df[col] = existing_sentiment_df[col].astype(dtype_val)
    
    # --- 1. Filter News Data Based on UID Range and Determine Effective Existing Data ---
    target_news_df = processed_news_df[
        (processed_news_df['UID'] >= start_uid) & (processed_news_df['UID'] <= end_uid)
    ].copy()

    if target_news_df.empty:
        print(f"No news data found in the specified UID range ({start_uid}-{end_uid}). Nothing to process.")
        # If overwriting an empty target range, the existing file (potentially modified if range was cleared) should be saved.
        if overwrite:
            print(f"Overwrite mode: Saving {sentiment_output_path} even with no target news, to reflect potential range clearing.")
            try:
                # If existing_sentiment_df was filtered for overwrite, that filtered version should be saved.
                temp_effective_df_for_empty_target = existing_sentiment_df.copy()
                if not temp_effective_df_for_empty_target.empty: # Only filter if not empty
                    temp_effective_df_for_empty_target = temp_effective_df_for_empty_target[
                        (temp_effective_df_for_empty_target['UID'] < start_uid) | (temp_effective_df_for_empty_target['UID'] > end_uid)
                    ].copy()
                
                temp_effective_df_for_empty_target.sort_values(by='UID', inplace=True)
                if 'UID' in temp_effective_df_for_empty_target.columns and not temp_effective_df_for_empty_target.empty:
                    temp_effective_df_for_empty_target['UID'] = temp_effective_df_for_empty_target['UID'].astype(int)
                temp_effective_df_for_empty_target.to_csv(sentiment_output_path, index=False, date_format='%Y-%m-%d')
                print("Save successful (reflecting overwrite on empty target range).")
            except Exception as e:
                print(f"Error saving during overwrite with empty target news: {e}")
        return

    uids_to_process_df = pd.DataFrame()
    effective_existing_df = pd.DataFrame(columns=expected_cols) # Initialize with schema

    if overwrite:
        print("Overwrite mode enabled. Processing all UIDs in the specified range.")
        uids_to_process_df = target_news_df.copy()
        # effective_existing_df contains data *outside* the current processing range
        if not existing_sentiment_df.empty:
            effective_existing_df = existing_sentiment_df[
                (existing_sentiment_df['UID'] < start_uid) | (existing_sentiment_df['UID'] > end_uid)
            ].copy()
            print(f"Existing sentiments within UID range {start_uid}-{end_uid} will be replaced by new results.")
        else: # existing_sentiment_df was empty
            for col, dtype_val in dtype_spec.items(): # Ensure schema for empty df
                 effective_existing_df[col] = effective_existing_df[col].astype(dtype_val)
    else: # Not overwrite
        effective_existing_df = existing_sentiment_df.copy() # Starts as full content
        if not effective_existing_df.empty:
            # UIDs that have valid (non-NaN) sentiment
            uids_with_valid_sentiment = set(
                effective_existing_df.dropna(subset=['Sentiment'])['UID'].unique()
            )
            # Process UIDs in target range that are NOT in uids_with_valid_sentiment
            uids_to_process_df = target_news_df[~target_news_df['UID'].isin(uids_with_valid_sentiment)].copy()
            
            num_skipped = len(target_news_df[target_news_df['UID'].isin(uids_with_valid_sentiment)])
            print(f"Found {num_skipped} UIDs in target range ({start_uid}-{end_uid}) with existing valid sentiment. Skipping them.")
        else:
            uids_to_process_df = target_news_df.copy()
            print("No existing sentiment data found. Processing all UIDs in target range.")


    if uids_to_process_df.empty:
        print("No new UIDs to process in the specified range (either already processed with valid sentiment and not overwriting, or target range was empty/fully covered).")
        if overwrite : # If overwrite was true, effective_existing_df (which has the range cleared) must be saved.
            print(f"Saving {sentiment_output_path} to reflect overwrite of range {start_uid}-{end_uid} as no new items were processed within it.")
            effective_existing_df.sort_values(by='UID', inplace=True)
            if 'UID' in effective_existing_df.columns and not effective_existing_df.empty:
                 effective_existing_df['UID'] = effective_existing_df['UID'].astype(int)
            try:
                effective_existing_df.to_csv(sentiment_output_path, index=False, date_format='%Y-%m-%d')
                print("Save successful (persisted overwrite of range).")
            except Exception as e:
                print(f"Error saving effective_existing_df after empty processing with overwrite: {e}")
        return

    total_to_process = len(uids_to_process_df)
    print(f"Total UIDs to process in this run: {total_to_process}")

    # --- 2. Process in Batches ---
    all_new_results = []
    num_batches = math.ceil(total_to_process / batch_size)

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, total_to_process)
        current_chunk_df = uids_to_process_df.iloc[start_index:end_index]


        print(f"\nProcessing batch {i+1}/{num_batches} (UIDs {current_chunk_df['UID'].min()} - {current_chunk_df['UID'].max()})...")

        max_retries = 2
        retries = 0
        batch_successful = False

        while retries <= max_retries and not batch_successful:
            try:
                batch_results_df = analyze_sentiment_wrapper(current_chunk_df, mode='batch', model_name=model_name)
                
                # Ensure batch_results_df has the same UIDs as current_chunk_df, or handle discrepancies
                if not batch_results_df['UID'].equals(current_chunk_df['UID'].reset_index(drop=True)):
                    print(f"Warning: UIDs in batch results do not perfectly match UIDs in the processed chunk for batch {i+1}. Aligning...")

                batch_successful = True
                all_new_results.append(batch_results_df)

            except Exception as e:
                retries += 1
                print(f"\nException encountered on batch {i+1} (Attempt {retries}/{max_retries + 1}). Error: {e}")
                if retries <= max_retries:
                    _save_current_progress(effective_existing_df, all_new_results, sentiment_output_path, overwrite)
                    wait_time = 60
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    print(f"Retrying batch {i+1}...")
                else:
                    print(f"Max retries ({max_retries}) exceeded for batch {i+1}. Skipping this batch.")
                    break


    # --- 3. Combine, Deduplicate, and Save ---
    if not all_new_results:
        print("No new sentiment results were generated.")
        return

    print("Combining new results with existing data...")
    new_results_df = pd.concat(all_new_results, ignore_index=True)

    # Combine effective_existing_df with new_results_df. New results take precedence.
    uids_in_new_results = new_results_df['UID'].unique()
    base_for_final_concat = effective_existing_df[~effective_existing_df['UID'].isin(uids_in_new_results)]
    
    final_sentiment_df = pd.concat([base_for_final_concat, new_results_df], ignore_index=True)

    # Sort by UID and ensure UID is int
    final_sentiment_df.sort_values(by='UID', inplace=True)
    if 'UID' in final_sentiment_df.columns and not final_sentiment_df.empty:
        # Ensure UID is Int64 before trying to convert to int, in case of pd.NA from merges/concats if a UID was missing
        final_sentiment_df['UID'] = final_sentiment_df['UID'].astype('Int64').astype(int)


    # --- 4. Save Results ---
    try:
        print(f"Saving {len(final_sentiment_df)} sentiment results to {sentiment_output_path}...")
        final_sentiment_df.to_csv(sentiment_output_path, index=False, date_format='%Y-%m-%d')
        print("Save successful.")
        max_processed_in_run = new_results_df['UID'].max() if not new_results_df.empty else start_uid -1 # Fallback if nothing new processed
        new_last_processed = max(last_processed_uid, max_processed_in_run, end_uid if uids_to_process_df.empty else -1) # Ensure requested range end is considered

        if new_last_processed > last_processed_uid:
             last_processed_uid = new_last_processed
             print(f"Updated global last_processed_uid to: {last_processed_uid}")


    except Exception as e:
        print(f"Error saving results to {sentiment_output_path}: {e}")

    print(f"--- Sentiment Processing Finished ---")


# %%
process_sentiment_by_uid(
        processed_news_df=all_processed_news_df,
        batch_size=10,
        start_uid=first_processed_uid,
        end_uid=last_processed_uid,
        overwrite=False,
        model_name="gemini-2.5-flash-preview-04-17",
        sentiment_output_path=sentiment_file_path
    )
print(f"\nCurrent last processed UID after run: {last_processed_uid}")

# %%
sentiment_df = pd.read_csv(sentiment_file_path, parse_dates=['Date'])
sentiment_df.tail(10)


