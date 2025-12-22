import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df: pd.DataFrame):
    """
    Fungsi preprocessing otomatis.
    Mengonversi hasil eksperimen notebook ke bentuk pipeline terstruktur.
    
    Parameter:
    ----------
    df : pd.DataFrame
        Dataset mentah (raw data)

    Return:
    -------
    df_processed : pd.DataFrame
        Dataset yang sudah dipreprocessing dan siap digunakan untuk training
    """

    # =====================================================
    # 1. Menghapus data duplikat
    # =====================================================
    df = df.drop_duplicates().reset_index(drop=True)

    # =====================================================
    # 2. Menangani missing values
    # (disesuaikan dengan eksperimen: drop missing)
    # =====================================================
    df = df.dropna().reset_index(drop=True)

    # =====================================================
    # 3. Identifikasi kolom numerik
    # =====================================================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # =====================================================
    # 4. Standardisasi fitur numerik
    # =====================================================
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # =====================================================
    # 5. Binning AccountBalance (sesuai eksperimen)
    # =====================================================
    if "AccountBalance" in df_scaled.columns:
        df_scaled["AccountBalance"] = pd.qcut(
            df_scaled["AccountBalance"],
            q=3,
            labels=["low", "mid", "high"]
        )

        # =================================================
        # 6. Encoding AccountBalance
        # =================================================
        label_encoder = LabelEncoder()
        df_scaled["AccountBalance"] = label_encoder.fit_transform(
            df_scaled["AccountBalance"]
        )

    # =====================================================
    # Output data
    # =====================================================
    return df_scaled
