
import numpy as np
import joblib
from preprocessor import preprocess
import pandas as pd

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load("best_classifier.pkl")
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(sample)
    labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    print(f"\nInput   : sepal=({sepal_length}, {sepal_width}), petal=({petal_length}, {petal_width})")
    print(f"Predicted Species: {labels.get(prediction[0], prediction[0])}")
    return prediction[0]

def main():
    print("=== Species Predictor ===")
    try:
        sl = float(input("Sepal Length: "))
        sw = float(input("Sepal Width : "))
        pl = float(input("Petal Length: "))
        pw = float(input("Petal Width : "))
        predict_species(sl, sw, pl, pw)
    except Exception as e:
        print(f"Error: {e}. Make sure you have trained the model first (python train_model.py)")

if __name__ == "__main__":
    main()
