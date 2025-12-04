"""
Configuration file for fraud detection model.
Contains all hyperparameters and settings.
"""

# Data settings
DATA_PATH = "synthetic_fraud_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model architecture
INPUT_SIZE = 13  # Number of features after preprocessing
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 256

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5

# Model save path
MODEL_SAVE_PATH = "../models/fraud_detection_model.pth"
