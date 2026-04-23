
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
from preprocessor import preprocess

def train_classifiers(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    print("\n--- Classification Results ---")
    print(f"{'Model':<25} {'Accuracy':>10} {'CV Score':>10}")
    print("-" * 50)
    best_model, best_score = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv  = cross_val_score(model, X_train, y_train, cv=5).mean()
        print(f"{name:<25} {acc*100:>9.1f}% {cv*100:>9.1f}%")
        if acc > best_score:
            best_score, best_model, best_name = acc, model, name
    print(f"\nBest Model: {best_name} ({best_score*100:.1f}%)")
    joblib.dump(best_model, "best_classifier.pkl")
    print("Model saved: best_classifier.pkl")
    return best_model

def train_regressor(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"\n--- Regression Results ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    joblib.dump(model, "linear_regressor.pkl")
    print("Model saved: linear_regressor.pkl")
    return model

def main():
    print("=== ML Model Training Pipeline ===")
    df = pd.read_csv("sample_dataset.csv")
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # Classification task
    X, y, le = preprocess(df, target="species", task="classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_classifiers(X_train, X_test, y_train, y_test)

    # Regression task (predict petal_length)
    X_reg, y_reg, _ = preprocess(df, target="petal_length", task="regression")
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    train_regressor(Xr_train, Xr_test, yr_train, yr_test)
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
