import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing dataset Heart Disease:
    - Encoding variabel biner
    - One-hot encoding variabel kategorikal
    - Menghasilkan dataset numerik siap modeling
    """

    # ===============================
    # 1. Drop duplicate & missing
    # ===============================
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # ===============================
    # 2. Encoding binary categorical
    # ===============================
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    # ===============================
    # 3. One-Hot Encoding categorical
    # ===============================
    cat_cols = ["ChestPainType", "RestingECG", "ST_Slope"]

    df_encoded = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    # ===============================
    # 4. Convert boolean → integer
    # ===============================
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    return df_encoded


if __name__ == "__main__":
    # ===============================
    # Path input & output
    # ===============================
    RAW_PATH = "../heart_raw.csv"   # sesuaikan nama file raw lo
    OUTPUT_PATH = "heart_preprocessing.csv"

    # ===============================
    # Run preprocessing
    # ===============================
    df_raw = pd.read_csv(RAW_PATH)
    df_processed = preprocess_data(df_raw)

    # ===============================
    # Save result
    # ===============================
    df_processed.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing selesai. File disimpan ke:", OUTPUT_PATH)