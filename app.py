# ============================================================
# CREDIT CARD FRAUD DETECTION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ============================================================
# 2. LOAD DATASET
# ============================================================

# If you already have creditcard.csv, you can directly use:
# df = pd.read_csv("creditcard.csv")

# For Google Colab + Kaggle:
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d mlg-ulb/creditcardfraud
# !unzip -o creditcardfraud.zip

df = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())


# ============================================================
# 4. VISUALIZE CLASS DISTRIBUTION
# ============================================================

df["Class"].value_counts().plot(
    kind="bar",
    title="Fraud vs Normal Transactions"
)

plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.show()


# ============================================================
# 5. CHECK DATA STATISTICS
# ============================================================

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 6. SCALE THE AMOUNT COLUMN
# ============================================================

# Amount can contain very large values.
# RobustScaler is less affected by outliers.

scaler = RobustScaler()

df["Amount"] = scaler.fit_transform(
    df[["Amount"]]
)


# ============================================================
# 7. SCALE THE TIME COLUMN
# ============================================================

# Convert Time approximately into a range of 0 to 1.

df["Time"] = (
    df["Time"] - df["Time"].min()
) / (
    df["Time"].max() - df["Time"].min()
)


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

# X = input features
# y = target/output

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nFraud distribution in training data:")
print(y_train.value_counts())

print("\nFraud distribution in testing data:")
print(y_test.value_counts())


# ============================================================
# 10. LOGISTIC REGRESSION
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train
logistic_model.fit(
    X_train,
    y_train
)

# Predict
logistic_predictions = logistic_model.predict(X_test)

# Evaluate
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Not Fraud", "Fraud"]
    )
)


# ============================================================
# 11. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

cm = confusion_matrix(
    y_test,
    logistic_predictions
)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Fraud", "Fraud"]
).plot()

plt.title("Logistic Regression Confusion Matrix")
plt.show()


# ============================================================
# 12. RANDOM FOREST
# ============================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(
    X_train,
    y_train
)

# Predict
rf_predictions = rf_model.predict(X_test)

# Evaluate
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Not Fraud", "Fraud"]
    )
)


# ============================================================
# 13. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    rf_predictions
)

print("\nConfusion Matrix:")
print(cm_rf)

ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Not Fraud", "Fraud"]
).plot()

plt.title("Random Forest Confusion Matrix")
plt.show()


# ============================================================
# 14. NEURAL NETWORK
# ============================================================

print("\n==============================")
print("NEURAL NETWORK")
print("==============================")

nn_model = Sequential([

    # Input + first hidden layer
    Dense(
        16,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    # Second hidden layer
    Dense(
        8,
        activation="relu"
    ),

    # Output layer
    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 15. COMPILE NEURAL NETWORK
# ============================================================

nn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Show architecture
nn_model.summary()


# ============================================================
# 16. TRAIN NEURAL NETWORK
# ============================================================

nn_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.20
)


# ============================================================
# 17. NEURAL NETWORK PREDICTIONS
# ============================================================

# Neural network gives probabilities
probabilities = nn_model.predict(X_test)

# Convert probabilities to 0/1
nn_predictions = (
    probabilities > 0.5
).astype(int).flatten()


# ============================================================
# 18. EVALUATE NEURAL NETWORK
# ============================================================

print("\nNeural Network Results:")

print(
    classification_report(
        y_test,
        nn_predictions,
        target_names=["Not Fraud", "Fraud"]
    )
)


# ============================================================
# 19. CONFUSION MATRIX - NEURAL NETWORK
# ============================================================

cm_nn = confusion_matrix(
    y_test,
    nn_predictions
)

print("\nConfusion Matrix:")
print(cm_nn)

ConfusionMatrixDisplay(
    confusion_matrix=cm_nn,
    display_labels=["Not Fraud", "Fraud"]
).plot()

plt.title("Neural Network Confusion Matrix")
plt.show()


# ============================================================
# 20. BALANCED DATASET
# ============================================================

print("\n==============================")
print("BALANCED DATASET")
print("==============================")


# Separate fraud and normal transactions

fraud = df[df["Class"] == 1]

normal = df[df["Class"] == 0]

print("\nNumber of fraud transactions:")
print(len(fraud))

print("\nNumber of normal transactions:")
print(len(normal))


# Take the same number of normal transactions
# as fraud transactions.

normal_sample = normal.sample(
    n=len(fraud),
    random_state=42
)


# Combine both datasets

balanced_df = pd.concat(
    [fraud, normal_sample]
)


# Shuffle

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
)


# Check distribution

print("\nBalanced dataset:")
print(
    balanced_df["Class"].value_counts()
)


# ============================================================
# 21. SEPARATE BALANCED DATA
# ============================================================

X_balanced = balanced_df.drop(
    "Class",
    axis=1
)

y_balanced = balanced_df["Class"]


# ============================================================
# 22. SPLIT BALANCED DATA
# ============================================================

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.20,
    random_state=42,
    stratify=y_balanced
)


print("\nBalanced training data:")
print(X_train_b.shape)

print("\nBalanced testing data:")
print(X_test_b.shape)


# ============================================================
# 23. LOGISTIC REGRESSION ON BALANCED DATA
# ============================================================

balanced_logistic = LogisticRegression(
    max_iter=1000
)

balanced_logistic.fit(
    X_train_b,
    y_train_b
)

balanced_predictions = balanced_logistic.predict(
    X_test_b
)


print("\nBalanced Logistic Regression:")

print(
    classification_report(
        y_test_b,
        balanced_predictions,
        target_names=["Not Fraud", "Fraud"]
    )
)


# ============================================================
# 24. RANDOM FOREST ON BALANCED DATA
# ============================================================

balanced_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

balanced_rf.fit(
    X_train_b,
    y_train_b
)

balanced_rf_predictions = balanced_rf.predict(
    X_test_b
)


print("\nBalanced Random Forest:")

print(
    classification_report(
        y_test_b,
        balanced_rf_predictions,
        target_names=["Not Fraud", "Fraud"]
    )
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")

print("""
Models used:

1. Logistic Regression
2. Random Forest
3. Neural Network

Main preprocessing:

1. RobustScaler for Amount
2. Min-Max scaling for Time

Main evaluation metrics:

1. Precision
2. Recall
3. F1-score
4. Confusion Matrix

Important problem:

The dataset is highly imbalanced because
fraudulent transactions are much fewer than
normal transactions.

We handled this using:

1. class_weight='balanced'
2. undersampling of normal transactions
""")
