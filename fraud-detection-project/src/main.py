

import torch
from torch import nn

from data_preprocessing import preprocess_pipeline, calculate_pos_weight
from dataset import create_dataloader
from model import initialize_model
from train import train_model, evaluate_model, save_model
import config


def main():
    """Main function to run the complete fraud detection pipeline."""


    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, y_train = preprocess_pipeline(
        filepath=config.DATA_PATH,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    

    train_dataloader = create_dataloader(
        X_train_tensor, 
        y_train_tensor, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
    )
    print(f"DataLoader created with batch size: {config.BATCH_SIZE}")
    

    model, device = initialize_model(
        input_size=config.INPUT_SIZE,
        hidden_size_1=config.HIDDEN_SIZE_1,
        hidden_size_2=config.HIDDEN_SIZE_2
    )
    
    # Calculate positive weight for imbalanced dataset
    pos_weight_value = calculate_pos_weight(y_train, device)
    
    # Setup loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # Step 4: Train model
    print("\n[4/5] Training model...")
    print("-" * 60)
    model = train_model(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=config.NUM_EPOCHS
    )

    
    # Step 5: Evaluate model
    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(
        model=model,
        X_test_tensor=X_test_tensor,
        y_test_tensor=y_test_tensor,
        device=device,
        threshold=config.CLASSIFICATION_THRESHOLD
    )
    
    # Save model
    print("\nSaving model...")
    save_model(model, config.MODEL_SAVE_PATH)
    
   
    
    return model, metrics


if __name__ == "__main__":
    main()
