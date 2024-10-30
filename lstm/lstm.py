# Imports
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Function to process data
def processData(data: pd.DataFrame) -> pd.DataFrame:
    data['closing_return'] = data['close'].pct_change()
    data['target'] = (data['closing_return'].shift(-1) > 0).astype(int)  # Ensure target is an integer
    data['spread'] = data['high'] - data['low']
    data['closing_return'] = data['closing_return'].fillna(0)  # Avoid chained assignment warning
    return data

# Load and Clean Data
path = os.path.join('/content/drive/MyDrive/dataset/train.csv')  # Adjusted path for Colab
data = pd.read_csv(path)
data = processData(data)
data['closing_return'] = data['closing_return'].fillna(0)  # Avoid chained assignment warning

# Function to create input data for the neural network
def create_nn_data(data, batch_size=10, save_scaler=True):
    X = data.drop(columns=['target']).values
    y = data['target'].values
    y = y[:len(y) - len(y) % batch_size : batch_size]
    y = y.reshape(-1, 1, 1)

    X = X[:len(X) - len(X) % batch_size]
    
    scaler_path = os.path.join('/content/trained-models/scaler.joblib')  # Adjusted path for Colab
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # Create directory if it doesn't exist
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

    X = X.reshape(-1, batch_size, X.shape[1])
    
    return X, y

# Create neural network data
BATCH_SIZE = 5
X, y = create_nn_data(data, batch_size=BATCH_SIZE)

# Build the Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))  
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(LSTM(64, return_sequences=True))  
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Add another dropout layer
model.add(LSTM(64))  # Last LSTM layer
model.add(Dense(1, activation='sigmoid'))  

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=BATCH_SIZE)  # Increased epochs


# Save the model for later
model_path = os.path.join('/content/trained-models/neural-net.keras')  # Adjusted path for Colab
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
model.save(model_path)

# Evaluate out-of-sample performance
test = pd.read_csv(os.path.join('/content/drive/MyDrive/dataset/test.csv'))  # Adjusted path for Colab
test = processData(test)
actual = test['target'].copy()
row_id = test['row_id'].copy()
test = test.drop(columns=['row_id', 'target'])

# Prepare test data for predictions
test = test.values
scaler = joblib.load(os.path.join('/content/trained-models/scaler.joblib'))  # Adjusted path for Colab
test = scaler.transform(test)

# Create input sequences for testing
X_test = []
for i in range(len(test) - BATCH_SIZE):
    X_test.append(test[i:i+BATCH_SIZE])
X_test = np.array(X_test)

# Load the model and make predictions
model = load_model(model_path)
y_pred = model.predict(X_test)
y_pred = np.pad(y_pred.flatten(), (0, BATCH_SIZE), mode='constant', constant_values=0)

# Prepare submission DataFrame
submission = pd.DataFrame({'row_id': row_id, 'target': y_pred.flatten()})
submission['target'] = submission['target'].apply(lambda x: 1 if x > 0.5 else 0)

# Save submission to CSV
submission.to_csv(os.path.join('predictions.csv'), index=False)  # Adjusted path for Colab

# Save the model for later
model_path = os.path.join('/content/trained-models/neural-net.keras')  # Adjusted path for Colab
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
model.save(model_path)

# Evaluate out-of-sample performance
test = pd.read_csv(os.path.join('/content/drive/MyDrive/dataset/test.csv'))  # Adjusted path for Colab
test = processData(test)
actual = test['target'].copy()
row_id = test['row_id'].copy()
test = test.drop(columns=['row_id', 'target'])

# Prepare test data for predictions
test = test.values
scaler = joblib.load(os.path.join('/content/trained-models/scaler.joblib'))  # Adjusted path for Colab
test = scaler.transform(test)

# Create input sequences for testing
X_test = []
for i in range(len(test) - BATCH_SIZE):
    X_test.append(test[i:i+BATCH_SIZE])
X_test = np.array(X_test)

# Load the model and make predictions
model = load_model(model_path)
y_pred = model.predict(X_test)
y_pred = np.pad(y_pred.flatten(), (0, BATCH_SIZE), mode='constant', constant_values=0)

# Prepare submission DataFrame
submission = pd.DataFrame({'row_id': row_id, 'target': y_pred.flatten()})
submission['target'] = submission['target'].apply(lambda x: 1 if x > 0.5 else 0)

# Save submission to CSV
submission.to_csv(os.path.join('predictions.csv'), index=False)  # Adjusted path for Colab
