import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    # =========================================
    # 1. SET TRACKING URI (SQLite - LOCAL)
    # =========================================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # =========================================
    # 2. SET EXPERIMENT NAME
    # =========================================
    mlflow.set_experiment("Heart_Disease_Prediction_Experiment")

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # =========================================
    # 3. LOAD DATASET (PREPROCESSED)
    # =========================================
    data_path = "heart_preprocessing.csv"

    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    # =========================================
    # 4. TRAIN TEST SPLIT
    # =========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================================
    # 5. START MLFLOW RUN
    # =========================================
    with mlflow.start_run(run_name="RF_HeartDisease_Baseline"):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # =====================================
        # 6. EVALUATION
        # =====================================
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("-" * 40)
        print("Model Berhasil Dilatih!")
        print("Accuracy:", acc)
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("-" * 40)
        print("Cek MLflow Dashboard Anda.")

if __name__ == "__main__":
    main()