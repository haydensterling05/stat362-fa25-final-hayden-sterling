import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization, Multiply, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, Huber
from keras.metrics import RootMeanSquaredError
from keras.regularizers import l2

# Function returning the compiled model
def stock_prediction_model(X_train, l2_reg = 0.001, dropout_reg = 0.3):

  # Defining the input parameters
  seq_len  = X_train.shape[1]
  n_features = X_train.shape[2]

  # 1. Input layer
  inputs = Input(shape=(seq_len, n_features))

  # 2. LSTM layer with regularization - 256 nodes
  x = LSTM(256, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(inputs)
  x = Dropout(dropout_reg)(x)

  # 3. LSTM layer with regularization - 128 nodes
  x = LSTM(128, return_sequences=False, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(x)
  x = Dropout(dropout_reg)(x)

  # 4. Dense layer with relu activation and regularization - 64 nodes
  x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
  x = Dropout(dropout_reg)(x)

  # 5. Output layer with linear activation for regression - 9 nodes for 9 ETF outputs
  outputs = Dense(9, activation="linear")(x)

  # Defining the model object
  model = Model(inputs, outputs)

  # Compiling the model
  model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=Huber(),
    metrics=[RootMeanSquaredError(name="rmse")]
  )

  return model

model = stock_prediction_model(X_train)

model.summary()