# %% [markdown]
# # Sentiment Analysis Visualization for 2022
# 
# This notebook visualizes sentiment data from 2022, exploring different aspects
# like daily sentiment trends, news volume, unique tickers, and sentiment distributions.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import calendar

# Set styling
plt.style.use('ggplot')
sns.set_palette('viridis')

# %% [markdown]
# ## Data Preparation
# First, let's prepare the data for visualization
sentiment_file_path_gemini = 'data/sentiment/news_sentiment_results_gemini.csv'
sentiment_df_gemini = pd.read_csv(sentiment_file_path_gemini)
sentiment_df_gemini['Date'] = pd.to_datetime(sentiment_df_gemini['Date'], errors='coerce')
sentiment_2022_df = sentiment_df_gemini[sentiment_df_gemini['Date'].dt.year == 2022].copy() # Use .copy() to avoid SettingWithCopyWarning
# %%
# Ensure Date column is datetime

# Convert sentiment to numeric if it's not already
# Assuming sentiment is stored as text or in a format that needs conversion
if sentiment_2022_df['Sentiment'].dtype == 'object':
    # Try to convert strings like "positive", "negative", "neutral" to numbers
    sentiment_map = {
        "Strongly Bearish": -3, "Bearish": -2, "Slightly Bearish": -1,
        "Neutral": 0,
        "Slightly Bullish": 1, "Bullish": 2, "Strongly Bullish": 3
    }
    # Use .loc to modify the DataFrame safely
    sentiment_2022_df.loc[:, 'Sentiment_Value'] = sentiment_2022_df['Sentiment'].map(sentiment_map)
    # Handle potential conversion failures if the map doesn't cover all cases
    sentiment_2022_df.loc[:, 'Sentiment_Value'] = pd.to_numeric(sentiment_2022_df['Sentiment_Value'], errors='coerce')

elif pd.api.types.is_numeric_dtype(sentiment_2022_df['Sentiment']):
     # If it's already numeric, just assign it
    sentiment_2022_df.loc[:, 'Sentiment_Value'] = sentiment_2022_df['Sentiment']
else:
    # Attempt conversion if it's not object or numeric (e.g., could be string numbers)
    sentiment_2022_df.loc[:, 'Sentiment_Value'] = pd.to_numeric(sentiment_2022_df['Sentiment'], errors='coerce')

# Drop rows where sentiment conversion failed
sentiment_2022_df.dropna(subset=['Sentiment_Value'], inplace=True)


# %% [markdown]
# ## Basic Analysis 1: Daily Trends (2x2 subplot)
# Let's visualize key daily metrics using the original sentiment scale (-3 to 3)

# %%
# Create daily aggregations using the original sentiment scale
daily_data = sentiment_2022_df.groupby(sentiment_2022_df['Date'].dt.date).agg(
    avg_sentiment=('Sentiment_Value', 'mean'),
    news_count=('UID', 'count'),
    unique_tickers=('Ticker', 'nunique'),
    sentiment_std=('Sentiment_Value', 'std') # Standard deviation of original sentiment
).reset_index()

# Convert back to datetime for better plotting
daily_data['Date'] = pd.to_datetime(daily_data['Date'])

# Create the 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Daily Sentiment Metrics for 2022 (Original Scale)', fontsize=16)

# Plot 1: Average Daily Sentiment
axes[0, 0].plot(daily_data['Date'], daily_data['avg_sentiment'], color='blue', linewidth=1.5)
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Neutral (0)') # Neutral line at 0
axes[0, 0].set_title('Average Daily Sentiment')
axes[0, 0].set_ylabel('Sentiment Score (-3 to 3)')
axes[0, 0].set_ylim(-1.5, 1.5) # Set explicit y-axis limits
axes[0, 0].legend()
axes[0, 0].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Plot 2: Daily News Count
axes[0, 1].plot(daily_data['Date'], daily_data['news_count'], color='green', linewidth=1.5)
axes[0, 1].set_title('Daily News Volume')
axes[0, 1].set_ylabel('Number of News Items')
axes[0, 1].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Plot 3: Daily Unique Tickers
axes[1, 0].plot(daily_data['Date'], daily_data['unique_tickers'], color='purple', linewidth=1.5)
axes[1, 0].set_title('Daily Unique Tickers')
axes[1, 0].set_ylabel('Number of Unique Tickers')
axes[1, 0].set_xlabel('Date')
axes[1, 0].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Plot 4: Daily Sentiment Volatility (Standard Deviation)
axes[1, 1].plot(daily_data['Date'], daily_data['sentiment_std'], color='red', linewidth=1.5)
axes[1, 1].set_title('Daily Sentiment Volatility (Std Dev)')
axes[1, 1].set_ylabel('Standard Deviation of Sentiment')
axes[1, 1].set_xlabel('Date')
axes[1, 1].xaxis.set_major_formatter(DateFormatter('%b %Y'))

# Improve layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## Basic Analysis 2: Monthly Trends
# Let's examine how sentiment changes by month throughout 2022

# %%
# Extract month from date
sentiment_2022_df['Month'] = sentiment_2022_df['Date'].dt.month

# Create monthly aggregations
monthly_data = sentiment_2022_df.groupby('Month').agg(
    avg_sentiment=('Sentiment_Value', 'mean'),
    news_count=('UID', 'count'),
    unique_tickers=('Ticker', 'nunique')
).reset_index()

# Convert month numbers to month names
monthly_data['Month_Name'] = monthly_data['Month'].apply(lambda x: calendar.month_abbr[x])

# Create figure
plt.figure(figsize=(12, 6))

# Plot average sentiment by month
ax1 = plt.gca()
ax1.bar(monthly_data['Month_Name'], monthly_data['avg_sentiment'], color='blue', alpha=0.6)
ax1.set_ylabel('Average Sentiment (-3 to 3)', color='blue') # Updated label
ax1.tick_params(axis='y', labelcolor='blue')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5) # Neutral line clearer

# Create second y-axis for news count
ax2 = ax1.twinx()
ax2.plot(monthly_data['Month_Name'], monthly_data['news_count'], color='red', marker='o', linewidth=2)
ax2.set_ylabel('News Count', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add title and adjust layout
plt.title('Monthly Sentiment and News Volume in 2022')
plt.grid(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ticker Analysis
# Let's analyze which tickers have the most news and their sentiment distributions

# %%
# Count news items per ticker
ticker_counts = sentiment_2022_df['Ticker'].value_counts().reset_index()
ticker_counts.columns = ['Ticker', 'News_Count']

# Get average sentiment per ticker
ticker_sentiment = sentiment_2022_df.groupby('Ticker')['Sentiment_Value'].mean().reset_index()
ticker_sentiment.columns = ['Ticker', 'Avg_Sentiment']

# Merge the two dataframes
ticker_data = pd.merge(ticker_counts, ticker_sentiment, on='Ticker')

# Get top 10 tickers by news count
top_tickers = ticker_data.sort_values('News_Count', ascending=False).head(10)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 10 tickers by news count
top_tickers_sorted = top_tickers.sort_values('News_Count')
ax1.barh(top_tickers_sorted['Ticker'], top_tickers_sorted['News_Count'], color='skyblue')
ax1.set_title('Top 10 Tickers by News Volume')
ax1.set_xlabel('Number of News Items')

# Plot 2: Sentiment for top 10 tickers
sentiment_colors = top_tickers_sorted['Avg_Sentiment'].apply(
    lambda x: 'forestgreen' if x > 0.1 else ('lightcoral' if x < -0.1 else 'grey')) # Adjusted colors slightly
ax2.barh(top_tickers_sorted['Ticker'], top_tickers_sorted['Avg_Sentiment'], color=sentiment_colors)
ax2.set_title('Average Sentiment for Top 10 Tickers')
ax2.set_xlabel('Average Sentiment Score (-3 to 3)') # Updated label
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5) # Neutral line clearer
ax2.set_xlim(min(top_tickers_sorted['Avg_Sentiment'].min() - 0.5, -1), # Adjust xlim dynamically
             max(top_tickers_sorted['Avg_Sentiment'].max() + 0.5, 1))


plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sentiment Distribution Analysis

# %%
# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sentiment Distribution Analysis (Original Scale)', fontsize=16) # Updated title

# Plot 1: Overall sentiment distribution
sns.histplot(sentiment_2022_df['Sentiment_Value'], bins=np.arange(-3.5, 4.5, 1), kde=False, ax=axes[0, 0]) # Use discrete bins
axes[0, 0].set_title('Distribution of Sentiment Scores')
axes[0, 0].set_xlabel('Sentiment Score (-3 to 3)') # Updated label
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1)) # Ensure integer ticks

# Plot 2: Sentiment distribution by month (boxplot)
month_order = [calendar.month_abbr[i] for i in range(1, 13)]
sns.boxplot(x=sentiment_2022_df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x]),
            y=sentiment_2022_df['Sentiment_Value'],
            ax=axes[0, 1], order=month_order) # Ensure correct month order
axes[0, 1].set_title('Sentiment Distribution by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Sentiment Score (-3 to 3)') # Updated label
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5) # Add neutral line

# Plot 3: Day of week effect
sentiment_2022_df['Day_of_Week'] = sentiment_2022_df['Date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg = sentiment_2022_df.groupby('Day_of_Week')['Sentiment_Value'].mean().reindex(day_order)
axes[1, 0].bar(day_avg.index, day_avg.values, color='purple', alpha=0.7)
axes[1, 0].set_title('Average Sentiment by Day of Week')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Average Sentiment (-3 to 3)') # Updated label
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5) # Add neutral line
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right') # Improve label rotation

# Plot 4: Rolling average sentiment (7-day window)
# Note: daily_data['avg_sentiment'] now uses the original scale
rolling_sentiment = daily_data.set_index('Date')['avg_sentiment'].rolling(window=7).mean()
axes[1, 1].plot(rolling_sentiment.index, rolling_sentiment.values, color='green', linewidth=2)
axes[1, 1].set_title('7-Day Rolling Average Sentiment')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Rolling Avg Sentiment (-3 to 3)') # Updated label
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5) # Add neutral line
axes[1, 1].xaxis.set_major_formatter(DateFormatter('%b %Y'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## Correlation Analysis: Sentiment vs. News Volume

# %%
# Calculate weekly aggregations
sentiment_2022_df['Week'] = sentiment_2022_df['Date'].dt.isocalendar().week
weekly_data = sentiment_2022_df.groupby('Week').agg(
    avg_sentiment=('Sentiment_Value', 'mean'),
    news_count=('UID', 'count')
).reset_index()

# Compute correlation
correlation = weekly_data['avg_sentiment'].corr(weekly_data['news_count'])

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(weekly_data['news_count'], weekly_data['avg_sentiment'],
           alpha=0.7, s=50, c=weekly_data['Week'], cmap='viridis') # Store scatter object

# Add trend line
z = np.polyfit(weekly_data['news_count'], weekly_data['avg_sentiment'], 1)
p = np.poly1d(z)
x_range = np.linspace(weekly_data['news_count'].min(), weekly_data['news_count'].max(), 100)
plt.plot(x_range, p(x_range), "r--", alpha=0.8)

# Add correlation text
plt.annotate(f"Correlation: {correlation:.2f}",
             xy=(0.05, 0.95),
             xycoords='axes fraction',
             fontsize=12, # Make annotation slightly larger
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.title('Weekly Correlation: Sentiment vs. News Volume')
plt.xlabel('Number of News Items per Week')
plt.ylabel('Average Weekly Sentiment (-3 to 3)') # Updated label
plt.colorbar(scatter, label='Week of Year') # Pass scatter object to colorbar
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Heatmap: Sentiment by Month and Day of Week

# %%
# Create month and day of week columns if they don't exist
if 'Month' not in sentiment_2022_df.columns:
    sentiment_2022_df['Month'] = sentiment_2022_df['Date'].dt.month
if 'Day_of_Week' not in sentiment_2022_df.columns:
    sentiment_2022_df['Day_of_Week'] = sentiment_2022_df['Date'].dt.dayofweek

# Calculate average sentiment for each month-day combination
heatmap_data = sentiment_2022_df.groupby(['Month', 'Day_of_Week'])['Sentiment_Value'].mean().unstack()

# Convert month numbers to names and day numbers to names
month_names = [calendar.month_abbr[i] for i in range(1, 13)]
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] # Use abbreviations

# Reindex heatmap data to ensure correct order
heatmap_data = heatmap_data.reindex(index=range(1, 13), columns=range(7))

# Create heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, # Center color map at neutral 0
                annot=True, fmt='.2f', linewidths=.5,
                xticklabels=day_names, yticklabels=month_names,
                annot_kws={"size": 10}) # Adjust annotation size
plt.title('Average Sentiment by Month and Day of Week')
plt.xlabel('Day of Week') # Add x-axis label
plt.ylabel('Month')      # Add y-axis label
plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Signal Count by Category
# Let's count the total number of signals generated for each sentiment category in 2022.

# %%
# Calculate counts for each sentiment category
signal_counts = sentiment_2022_df['Sentiment'].value_counts()

# Display the counts
print("Total Signal Counts by Category for 2022:")
print(signal_counts)

# Optional: Define the desired order for categories if needed
category_order = [
    "Strongly Bullish", "Bullish", "Slightly Bullish",
    "Neutral",
    "Slightly Bearish", "Bearish", "Strongly Bearish"
]

# Optional: Plot the counts as a bar chart
plt.figure(figsize=(10, 5))
try:
    # Try to reindex based on the defined order
    signal_counts.reindex(category_order).plot(kind='bar', color='teal', alpha=0.7)
except KeyError:
    # If some categories are missing in the data, plot as is
    signal_counts.plot(kind='bar', color='teal', alpha=0.7)

plt.title('Total Signal Count by Sentiment Category in 2022')
plt.ylabel('Number of Signals')
plt.xlabel('Sentiment Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion
# 
# These visualizations provide various perspectives on the sentiment data from 2022, using the original -3 to 3 sentiment scale for better interpretability:
# 
# 1. We can see daily trends in sentiment (centered around 0), news volume, and ticker diversity
# 2. Monthly patterns reveal potential seasonal effects on market sentiment
# 3. The ticker analysis shows which companies received the most attention and their average sentiment profiles
# 4. The distribution analysis helps understand the overall sentiment landscape, showing the prevalence of different score levels
# 5. The correlation analysis explores the relationship between news volume and sentiment levels
# 6. The heatmap reveals patterns in sentiment by day of week and month, centered around neutral
# 
# Further analysis could explore:
# - Sentiment impact on stock price movements
# - Topic modeling of reasons for sentiment
# - Predictive modeling using sentiment as a feature