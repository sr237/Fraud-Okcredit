#  Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification algorithms. The project focuses on handling the highly imbalanced nature of fraud detection datasets and comparing different machine learning models.

---

##  Project Overview

Credit card fraud detection is a **binary classification problem** where each transaction is classified as either:

- `0` → Normal transaction
- `1` → Fraudulent transaction

The dataset contains **284,807 transactions**, out of which only **492 are fraudulent**.

This severe class imbalance makes fraud detection challenging because a model can achieve very high accuracy simply by predicting most transactions as normal.

Therefore, this project focuses on:

- Precision
- Recall
- F1-score
- Confusion Matrix

rather than relying only on accuracy.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset** from Kaggle.

The dataset contains:

| Feature | Description |
|--------|-------------|
| `Time` | Time elapsed between transactions |
| `V1 - V28` | Anonymized transaction features obtained using PCA |
| `Amount` | Transaction amount |
| `Class` | Target variable |

### Target Variable

```text
Class = 0 → Normal Transaction
Class = 1 → Fraudulent Transaction
