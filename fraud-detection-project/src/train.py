"""
Training and evaluation functions for fraud detection model.
"""

import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support


def train_model(model, dataloader, loss_fn, optimizer, device, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass (raw logits)
            outputs = model(batch_X)
            
            # Compute loss
            loss = loss_fn(outputs.squeeze(), batch_y.float())
            
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)  # Accumulate loss
        
        epoch_loss /= len(dataloader.dataset)  # Average loss over dataset
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.10f}")
    
    return model


def evaluate_model(model, X_test_tensor, y_test_tensor, device, threshold=0.5):

    # Set model to evaluation mode
    model.eval()
    
    with torch.inference_mode():
        test_logits = model(X_test_tensor.to(device)).squeeze()
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > threshold).int()
    
    # Move targets to device and ensure they are int type
    y_test_int = y_test_tensor.to(device).int()
    
    # Calculate accuracy
    correct_predictions = (test_preds == y_test_int).sum().item()
    total_predictions = len(y_test_int)
    accuracy = correct_predictions / total_predictions
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_int.cpu().numpy(), 
        test_preds.cpu().numpy(), 
        average='binary'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    return metrics


def save_model(model, filepath):

    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):

    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model
