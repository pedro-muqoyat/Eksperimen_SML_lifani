import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

# 1. Alamat Server Remote DagsHub
DAGSHUB_REPO_OWNER = "pedro-muqoyat"
DAGSHUB_REPO_NAME = "Eksperimen_SML_lifani"
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Baseline_Model_Student_Dropout")

def main():
    print("[1/5] Membaca file dataset.csv di folder lokal...")
    if not os.path.exists("dataset.csv"):
        print("[GALAT] File dataset.csv tidak ditemukan di folder ini! Pastikan letaknya sejajar dengan modelling.py.")
        return

    df = pd.read_csv("dataset.csv")
    
    # Mengubah target text (Dropout, Enrolled, Graduate) menjadi angka (0, 1, 2)
    target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if df["Target"].dtype == object:
        df["Target"] = df["Target"].map(target_mapping)
        
    X = df.drop(columns=["Target"])
    y = df["Target"]
    
    # Membagi data: 80% latihan, 20% ujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[2/5] Memulai sesi pencatatan ke DagsHub MLflow...")
    with mlflow.start_run(run_name="RandomForest_Local_Run"):
        
        # Parameter model yang kita tentukan
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        
        # Mencatat parameter ke MLflow secara manual/eksplisit (Syarat Nilai Tinggi)
        for kunci, nilai in params.items():
            mlflow.log_param(kunci, nilai)
            
        print("[3/5] Melatih model di komputer lokal Anda...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Menguji model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        # Mencatat metrik hasil ujian ke MLflow DagsHub
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        
        print("\n=== HASIL EVALUASI MODEL ===")
        print(f"Akurasi Model: {acc:.4f}")
        print(f"F1-Score Model: {f1:.4f}\n")
        
        print("[4/5] Mengirimkan file model (.pkl) ke server DagsHub...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name="Model_Mahasiswa_RandomForest")
        
        print("[5/5] Selesai! Semua data latihan telah terunggah ke DagsHub.")

if __name__ == "__main__":
    main()