# 🛡️ Fraud Detection with PyTorch

A modular, production-ready Deep Learning project for detecting fraudulent transactions. This project demonstrates best practices in structuring PyTorch applications, including custom Datasets, modular preprocessing, and separated training logic.

## 📂 Project Structure

The project is organized to separate concerns between data, modeling, and training:

```text
fraud-detection-project/
├── data/                      # 💾 Store your 'synthetic_fraud_dataset.csv' here
├── models/                    # 🤖 Saved model checkpoints (.pth files)
├── notebooks/                 # 📓 Jupyter notebooks for EDA and prototyping
├── src/                       # 🧠 Source code
│   ├── config.py              #    Hyperparameters & file paths
│   ├── data_preprocessing.py  #    Cleaning, encoding, and splitting logic
│   ├── dataset.py             #    Custom PyTorch Dataset & DataLoader
│   ├── model.py               #    Neural Network architecture
│   ├── train.py               #    Training loop & evaluation functions
│   ├── utils.py               #    Visualization & helper functions
│   ├── main.py                #    🚀 Entry point for training pipeline
│   └── predict.py             #    🔮 Script for making predictions
├── requirements.txt           # 📦 Project dependencies
└── README.md                  # 📄 This file
```

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.8+
*   PyTorch 2.0+

### 2. Installation

Clone the repository and install the required packages:

```bash
# Navigate to the project directory
cd fraud-detection-project

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Setup

1.  Download or prepare your dataset (expected filename: `synthetic_fraud_dataset.csv`).
2.  Place the CSV file inside the `data/` folder.

## 🏃‍♂️ Usage

### Training the Model

To run the full training pipeline (preprocessing -> training -> evaluation -> saving):

```bash
cd src
python main.py
```

**What happens:**
*   Data is loaded and cleaned.
*   Categorical features are one-hot encoded.
*   Data is split into Train/Test sets.
*   The model trains for the epochs defined in `config.py`.
*   The best model is saved to `models/fraud_detection_model.pth`.

### Making Predictions

To use the trained model for inference on new data:

```bash
cd src
python predict.py
```

## ⚙️ Configuration

You can adjust hyperparameters in `src/config.py` without touching the core code:

*   **`BATCH_SIZE`**: 64
*   **`LEARNING_RATE`**: 0.01
*   **`NUM_EPOCHS`**: 10
*   **`HIDDEN_SIZE`**: Adjust layer dimensions
*   **`CLASSIFICATION_THRESHOLD`**: 0.5 (Tune this for precision/recall trade-off)

## 🧠 Model Architecture

The project uses a fully connected Feed-Forward Neural Network (FNN):

*   **Input Layer**: 13 features (after encoding)
*   **Hidden Layer 1**: 128 neurons + ReLU
*   **Hidden Layer 2**: 256 neurons + ReLU
*   **Output Layer**: 1 neuron (Logits)

*Note: We use `BCEWithLogitsLoss` which combines a Sigmoid layer and the BCELoss in one single class. This is more numerically stable than using a plain Sigmoid followed by a BCELoss.*

## 📊 Handling Imbalanced Data

Fraud datasets are typically highly imbalanced (very few fraud cases). This project handles this by:
1.  **Stratified Splitting**: Ensuring the train/test split maintains the same ratio of fraud cases.
2.  **Weighted Loss**: Calculating `pos_weight` for the loss function to penalize the model more for missing a fraud case.

## 📈 Results

After training, the script outputs:
*   **Accuracy**
*   **Precision**
*   **Recall**
*   **F1-Score**

Check the console output for the confusion matrix and ROC curve details.

## 🤝 Contributing

Feel free to fork this project and submit PRs.

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)
