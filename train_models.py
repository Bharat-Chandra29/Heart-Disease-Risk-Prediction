import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARNINGS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Create plots folder if not exists
os.makedirs("plots", exist_ok=True)

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load("dataset.pkl")

# Convert to dense format (for pandas)
X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# Generate feature column names
from joblib import load
original_df = pd.read_csv("heart.csv")
X = original_df.drop(columns=["HeartDisease", "PhysicalHealth", "MentalHealth"])
feature_columns = pd.get_dummies(X, drop_first=True).columns.tolist()

# Create DataFrame for heatmap and target distribution
df_train = pd.DataFrame(X_train_dense, columns=feature_columns)
df_train['target'] = y_train

# ------------------ Improved Correlation Heatmap ------------------
plt.figure(figsize=(16, 14))
corr_matrix = df_train.drop(columns=["target"]).corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9}
)

plt.title("Feature Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.close()


# ------------------ Target Distribution ------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title("Target Class Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

# ------------------ Random Forest ------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")
joblib.dump(rf_model, "rf_model.pkl")

# ------------------ XGBoost ------------------
xgb_model = XGBClassifier(eval_metric='logloss', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
joblib.dump(xgb_model, "xgb_model.pkl")

# ------------------ Logistic Regression ------------------
log_model = LogisticRegression(max_iter=300, solver='liblinear')
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
joblib.dump(log_model, "log_model.pkl")

# ------------------ Feedforward Neural Network ------------------
ffnn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ffnn_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

ffnn_model.fit(X_train, y_train,
               epochs=20,
               batch_size=64,
               validation_data=(X_test, y_test),
               callbacks=[early_stop],
               verbose=0)

ffnn_acc = ffnn_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Feedforward Neural Network Accuracy: {ffnn_acc:.4f}")
ffnn_model.save("ffnn_model.keras", include_optimizer=False)

# ------------------ ROC Curve & Confusion Matrix for Logistic Regression ------------------
y_log_probs = log_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, y_log_probs)
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f'LogReg (AUC = {roc_auc_log:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_logistic_regression.png")
plt.close()

cm_log = confusion_matrix(y_test, log_model.predict(X_test))
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_logistic_regression.png")
plt.close()

# ------------------ ROC Curve & Confusion Matrix for FFNN ------------------
y_ffnn_probs = ffnn_model.predict(X_test).ravel()
fpr_ffnn, tpr_ffnn, _ = roc_curve(y_test, y_ffnn_probs)
roc_auc_ffnn = auc(fpr_ffnn, tpr_ffnn)

plt.figure()
plt.plot(fpr_ffnn, tpr_ffnn, label=f'FFNN (AUC = {roc_auc_ffnn:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - FFNN")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_ffnn.png")
plt.close()

y_ffnn_pred = (y_ffnn_probs > 0.5).astype(int)
cm_ffnn = confusion_matrix(y_test, y_ffnn_pred)
disp_ffnn = ConfusionMatrixDisplay(confusion_matrix=cm_ffnn)
disp_ffnn.plot()
plt.title("Confusion Matrix - FFNN")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_ffnn.png")
plt.close()

# ------------------ Feature Importance - Logistic Regression ------------------
importance = abs(log_model.coef_[0])
indices = np.argsort(importance)[::-1]
top_n = 15

plt.figure(figsize=(10, 6))
plt.bar(range(top_n), importance[indices[:top_n]])
plt.xticks(range(top_n), np.array(feature_columns)[indices[:top_n]], rotation=45, ha='right')
plt.title("Top 15 Feature Importances (Logistic Regression)")
plt.tight_layout()
plt.savefig("plots/feature_importance_logistic_regression.png")
plt.close()