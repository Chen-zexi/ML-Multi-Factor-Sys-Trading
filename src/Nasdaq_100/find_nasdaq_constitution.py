# %%
import json

# Define the input and output file paths
input_json_path = '../../data/nasdaq_100/nasdaq_100_members_2015_2024.json'
output_json_path = '../../data/nasdaq_100/unique_nasdaq_tickers.json'

# Load the JSON data from the input file
with open(input_json_path, 'r') as f:
    data = json.load(f)

# Use a set to store unique tickers
unique_tickers = set()

# Iterate through the values (lists of tickers) in the loaded data
for key in data:
    if isinstance(data[key], list):
        unique_tickers.update(data[key])

# Convert the set back to a list (optional, but common for JSON)
unique_tickers_list = sorted(list(unique_tickers))

# Save the unique tickers to the output JSON file
with open(output_json_path, 'w') as f:
    json.dump(unique_tickers_list, f, indent=4)

print(f"Extracted {len(unique_tickers_list)} unique tickers.")
print(f"Unique tickers saved to '{output_json_path}'")
