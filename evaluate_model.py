
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from preprocessor import preprocess

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        fi_df = fi_df.sort_values("Importance", ascending=False)
        plt.figure(figsize=(7, 4))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        print("Saved: feature_importance.png")
        plt.close()

def evaluate():
    df = pd.read_csv("sample_dataset.csv")
    X, y, le = preprocess(df, target="species", task="classification")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load("best_classifier.pkl")
    y_pred = model.predict(X_test)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nClassification Report:")
    classes = le.classes_ if le else None
    print(classification_report(y_test, y_pred, target_names=classes))
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_feature_importance(model, X.columns.tolist())

if __name__ == "__main__":
    evaluate()
