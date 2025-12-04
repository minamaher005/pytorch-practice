"""
Example script showing how to make predictions with a trained model.
"""

import torch
from model import FraudDetectionModel, get_device
from train import load_model
from data_preprocessing import preprocess_pipeline
import config


def predict_single_transaction(model, transaction_features, device):

    model.eval()
    with torch.inference_mode():
        logit = model(transaction_features.to(device))
        probability = torch.sigmoid(logit).item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability


def main():
   
    # Get device
    device = get_device()
    
    # Initialize model architecture
    model = FraudDetectionModel(
        input_size=config.INPUT_SIZE,
        hidden_size_1=config.HIDDEN_SIZE_1,
        hidden_size_2=config.HIDDEN_SIZE_2
    )
    
    # Load trained weights
    model = load_model(model, config.MODEL_SAVE_PATH, device)
    
    # Example: Load test data
    X_train, X_test, y_train, y_test, scaler, _ = preprocess_pipeline(
        filepath=config.DATA_PATH,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # Predict on first test sample



if __name__ == "__main__":
    main()
