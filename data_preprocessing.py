# Importing the necessary packages
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


### Function to download the financial data from yfinance
def download_financial_data():

  # Define the sectors to use for the model
  sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']

  # Define the start and end dates of the model
  start = "2012-01-30"
  end = "2022-09-22"

  # Empty list to store each dataframe
  df_list = []

  # Downloading data for each sector in a dataframe
  for sector in sectors:
      ticker = yf.Ticker(sector)
      df_sector = ticker.history(start=start, end=end)
      df_sector = df_sector[['Open', 'High', 'Low', 'Close', 'Volume']]
      df_sector.index = df_sector.index.tz_localize(None)
      df_sector.columns = [f'{sector}_{col}' for col in df_sector.columns]
      df_list.append(df_sector)

  # Combine all dataframes
  df_financial = pd.concat(df_list, axis=1)

  # Printing some results
  print("\n---Financial Data Info---")
  print(f"Date range: {df_financial.index[0]} to {df_financial.index[-1]}")
  print(f"Total trading days: {df_financial.shape[0]}")
  print(f"Number of sectors in data: {len(sectors)}")
  print(f"Number of features per sector: {int(df_financial.shape[1] / len(sector))}")
  print(f"Total number of financial features: {df_financial.shape[1]}")

  return df_financial



### Function to download the raw news data
def download_news_data(path_to_data = "News_Category_Dataset_v3.json"):

  # Extracting all headlines
  news_data = pd.read_json(path_to_data, lines=True, nrows=1000000)

  # Defining the categories to keep from the data
  keep_categories = [
      'BUSINESS',       # Directly related to the stock market
      'MONEY',          # Directly related to the stock market
      'TECH',           # Tech news affects XLK heavily
      'WORLD NEWS',     # International events, trade, geopolitics
      'U.S. NEWS',      # Domestic events that can move markets
      'ENVIRONMENT',    # Climate policy, energy regulations (affects XLE, XLU)
      'SCIENCE'         # Biotech, pharma news (affects XLV)
  ]
  news_filtered = news_data[news_data['category'].isin(keep_categories)]

  # Converting the date column to datetime format
  news_filtered.loc[:, 'date'] = pd.to_datetime(news_filtered['date'])

  # Printing some results
  print("\n---Raw News Data Info---")
  print(f"Date range: {news_filtered['date'].min()} to {news_filtered['date'].max()}")
  print(f"Number of headlines: {news_filtered.shape[0]}")


  return news_filtered



### Function to extract news sentiments
def extract_news_sentiment(df_news):

  # Initiating the pretrained sentiment analyzer
  analyzer = SentimentIntensityAnalyzer()

  # Empty list to store sentiment features
  daily_sentiments = []

  # Group by date across all categories (no category separation)
  for date, group in df_news.groupby('date'):
      scores = []
      for headline in group['headline']:
          if isinstance(headline, str) and len(headline) > 0:
              sentiment = analyzer.polarity_scores(headline)
              scores.append(sentiment['compound'])
      if scores:
          daily_sentiments.append({
              'date': date,
              'sentiment': np.mean(scores),
              'sentiment_std': np.std(scores),
              'num_headlines': len(scores)
          })

  # Making a dataframe out of all the sentiment
  sentiment_df = pd.DataFrame(daily_sentiments)
  sentiment_df.set_index('date', inplace=True)

  # Imputing the missing sentiment values
  start_date = sentiment_df.index.min()
  end_date = sentiment_df.index.max()
  full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
  sentiment_complete = sentiment_df.reindex(full_date_range)
  for column in sentiment_complete.columns:
    # Calculate rolling mean from past 5 days
    rolling_mean = sentiment_complete[column].shift(1).rolling(window=5, min_periods=1).mean()
    # Fill missing values with the rolling mean
    sentiment_complete[column] = sentiment_complete[column].fillna(rolling_mean)
  sentiment_complete = sentiment_complete.ffill()
  sentiment_complete.index.name = 'date'

  # Printing some results
  print("\n---News Sentiments Data Info---")
  print(f"Date range: {sentiment_complete.index.min()} to {sentiment_complete.index.max()}")
  print(f"Extracted sentiment features: {sentiment_complete.columns.to_list()}")

  return sentiment_complete



### Function to merge the two datasets
def merge_datasets(df_financial, df_sentiments):

  # Merging the financial and sentiments data
  df = pd.merge(df_financial, df_sentiments, left_index=True, right_index=True)

  # Printing some results
  print("\n---Merged Data Info---")
  print(f"Date range: {df.index.min()} to {df.index.max()}")
  print(f"Number of trading days: {df.shape[0]}")
  print(f"Total number of features: {df.shape[1]}")

  return df
