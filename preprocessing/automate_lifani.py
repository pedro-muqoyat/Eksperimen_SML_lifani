import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset mentah ke dalam memori DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[GALAT FATAL] File tidak ditemukan pada rute: {file_path}")
    
    # \mathbf{X} \in \mathbb{R}^{B \times F} -> Pembacaan adaptif terhadap pemisah delimiter
    try:
        df = pd.read_csv(file_path, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=',')
        return df
    except Exception as e:
        raise RuntimeError(f"Gagal memuat dataset: {str(e)}")

def build_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Membangun arsitektur pra-pemrosesan data numerik dan kategorikal."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def run_automation():
    """Fungsi eksekusi alur otomatisasi data prep."""
    raw_data_path = "dataset.csv"
    output_dir = "../preprocessing"
    output_path = os.path.join(output_dir, "data_bersih.csv")

    print("[INFO] Memulai otomatisasi pra-pemrosesan dataset...")

    # 1. Ekstraksi Data
    df = load_data(raw_data_path)
    target_column = 'Target'
    
    # 2. Pemisahan Fitur dan Target
    y_raw = df[target_column]
    X = df.drop(columns=[target_column])
    
    # 3. Label Encoding Target 
    label_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    y_encoded = y_raw.map(label_mapping)

    # 4. Deteksi Tipe Data Kolom Dinamis
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 5. Fitting dan Transformasi Data
    print(f"[INFO] Mentransformasi {len(numeric_cols)} kolom numerik dan {len(categorical_cols)} kolom kategorikal...")
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_processed = preprocessor.fit_transform(X)
    
    # 6. Rekonstruksi DataFrame dengan Proteksi Bypass (SOLUSI)
    if len(categorical_cols) > 0:
        # Dieksekusi hanya jika matriks kategorikal \mathbf{X}_{cat} \neq \emptyset
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        new_cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
    else:
        # Bypass jika fitur kategorikal bernilai 0
        new_cat_cols = []
        
    all_cols = numeric_cols + new_cat_cols
    
    df_clean = pd.DataFrame(X_processed, columns=all_cols)
    df_clean[target_column] = y_encoded.values
    
    # 7. Penyimpanan Artefak
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[SUKSES] Dataset berhasil dibersihkan dan disimpan di: {output_path}")
    print(f"[METADATA] Dimensi Dataset Akhir: {df_clean.shape}")

if __name__ == "__main__":
    run_automation()