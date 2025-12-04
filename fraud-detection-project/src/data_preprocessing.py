

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def load_data(filepath):
    """Load the fraud dataset from CSV file."""
    data = pd.read_csv(filepath)
    return data


def clean_data(data):
    """Drop unnecessary columns from the dataset."""
    columns_to_drop = ['country', 'user_id', 'transaction_id']
    data_cleaned = data.drop(columns=columns_to_drop, axis=1)
    return data_cleaned


def encode_categorical_features(data):

    data_encoded = pd.get_dummies(
        data, 
        columns=['transaction_type', 'merchant_category']
    )
    
    # Convert boolean columns to integers
    bool_cols = data_encoded.select_dtypes('bool').columns
    data_encoded = data_encoded.astype({col: 'int' for col in bool_cols})
    
    return data_encoded


def split_and_normalize_data(data, test_size=0.2, random_state=42):

    # Separate features and target
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    # Split the data (stratify to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Normalize numeric columns
    numeric_cols = ['amount', 'hour', 'device_risk_score', 'ip_risk_score']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler


def convert_to_tensors(X, y):
    """Convert numpy arrays to PyTorch tensors."""
    X_numpy = X.values
    y_numpy = y.values
    
    X_tensor = torch.from_numpy(X_numpy).float()
    y_tensor = torch.from_numpy(y_numpy).float()
    
    return X_tensor, y_tensor


def calculate_pos_weight(y_train, device):
 
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight_value = torch.tensor(
        [neg_count / pos_count], 
        device=device, 
        dtype=torch.float32
    )
    print(f"Calculated positive weight: {pos_weight_value.item():.2f}")
    return pos_weight_value


def preprocess_pipeline(filepath, test_size=0.2, random_state=42):

    # Load and clean data
    data = load_data(filepath)
    
    # Encode categorical features
    data_encoded = encode_categorical_features(data)
    
    # Clean unnecessary columns
    data_cleaned = clean_data(data_encoded)
    
    # Split and normalize
    X_train, X_test, y_train, y_test, scaler = split_and_normalize_data(
        data_cleaned, test_size, random_state
    )
    
    # Convert to tensors
    X_train_tensor, y_train_tensor = convert_to_tensors(X_train, y_train)
    X_test_tensor, y_test_tensor = convert_to_tensors(X_test, y_test)
    
    print(f"Training set shape: {X_train_tensor.shape}")
    print(f"Test set shape: {X_test_tensor.shape}")
    print(f"\nTraining set 'is_fraud' distribution:\n{y_train.value_counts()}")
    print(f"\nTest set 'is_fraud' distribution:\n{y_test.value_counts()}")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, y_train
