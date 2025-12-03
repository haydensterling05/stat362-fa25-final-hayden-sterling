import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization, Multiply, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.regularizers import l2

# Defining the callbacks
callbacks = [

    # Early Stopping to reduce overfitting and restoring the weights from the best epoch
    EarlyStopping(
      monitor = "val_loss",
      patience = 10,
      restore_best_weights = True
    ),

    # Reduce learning rate when validation loss plateaus for more finely tuned optimization
    ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.5,
      patience=5,
      min_lr=1e-6,
      verbose=1
    )

]

# Training the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks,
    verbose=1
)
