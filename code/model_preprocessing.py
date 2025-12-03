# Importing necessary libraries
import numpy as np
import pandas as pd



### Function to do train-val-test split
def train_val_test_split(df):

  # Conducting the train-test split
  n = len(df)
  train_end = int(0.7 * n)
  val_end = int(0.85 * n)

  train = df.iloc[:train_end]
  val = df.iloc[train_end:val_end]
  test = df.iloc[val_end:]

  # Printing some results
  print("\n---Train-Val-Test Split Info---")
  print(f"Train size: {train.shape[0]}")
  print(f"Validation size: {val.shape[0]}")
  print(f"Test size: {test.shape[0]}")

  return train, val, test



### Function to do target standardization
def target_standardization(df, train, val, test):
  sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']

  # Apply 100-day rolling mean and std scaling for Close prices
  window_size = 100

  # Dictionary to store rolling statistics for each sector
  rolling_stats = {}

  # Calculating the rolling statistics
  for sector in sectors:

      # Extracting the columns to use
      close_col = f'{sector}_Close'
      full_close = df[close_col].copy()

      # Calculating
      rolling_mean = full_close.rolling(window=window_size, min_periods=1).mean()
      rolling_std = full_close.rolling(window=window_size, min_periods=1).std()
      rolling_std = rolling_std.fillna(1.0).replace(0, 1.0)

      # Store statistics for each split
      rolling_stats[sector] = {
          'mean_train': rolling_mean.loc[train.index],
          'std_train': rolling_std.loc[train.index],
          'mean_val': rolling_mean.loc[val.index],
          'std_val': rolling_std.loc[val.index],
          'mean_test': rolling_mean.loc[test.index],
          'std_test': rolling_std.loc[test.index]
      }

  # Scale Close prices for all datasets, copying first
  train_scaled = train.copy()
  val_scaled = val.copy()
  test_scaled = test.copy()

  # Standardizing the target column (Close) according to the 100-day rolling statistics
  for sector in sectors:
    close_col = f'{sector}_Close'
    train_scaled[close_col] = (train[close_col] - rolling_stats[sector]['mean_train']) / rolling_stats[sector]['std_train']
    val_scaled[close_col] = (val[close_col] - rolling_stats[sector]['mean_val']) / rolling_stats[sector]['std_val']
    test_scaled[close_col] = (test[close_col] - rolling_stats[sector]['mean_test']) / rolling_stats[sector]['std_test']

  # Printing some results
  print("\n---Target Standardization Info---")
  print(f"Train size: {train_scaled.shape[0]}")
  print(f"Validation size: {val_scaled.shape[0]}")
  print(f"Test size: {test_scaled.shape[0]}")
  print("Rolling statistics have been stored.")

  return train_scaled, val_scaled, test_scaled, rolling_stats



### Function to create sequences for training
def create_sequences(df, look_back = 60):
  sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']

  # Preparing for sequence creation
  values = df.values.astype(np.float32)
  columns = df.columns
  target_cols = [f'{sector}_Close' for sector in sectors]
  target_indices = [columns.get_loc(col) for col in target_cols]
  X, y = [], []

  # Creating sequences for the model
  for end_idx in range(look_back, len(df)):
      window = values[end_idx - look_back:end_idx].copy() # Training sequences include days T - look_back to T-1

      # Scale non-target features using per-window statistics
      for col_idx in range(window.shape[1]):
          if col_idx not in target_indices:  # Skip target columns (already scaled)
              col_mean = window[:, col_idx].mean()
              col_std = window[:, col_idx].std()
              if col_std == 0:
                  col_std = 1.0
              window[:, col_idx] = (window[:, col_idx] - col_mean) / col_std

      X.append(window)
      y.append([values[end_idx, idx] for idx in target_indices])  # Labels include Close price from day T

  # Converting to numpy arrays
  X = np.array(X)
  y = np.array(y)

  # Printing results
  print("\n---Sequences have been created---")
  print(f"X shape is: {X.shape}")
  print(f"y shape is: {y.shape}")

  return X, y
