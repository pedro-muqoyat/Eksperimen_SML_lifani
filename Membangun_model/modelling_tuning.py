import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. KONFIGURASI LINGKUNGAN & KREDENSIAL
# =====================================================================
DAGSHUB_USERNAME = "pedro-muqoyat"
DAGSHUB_REPO = "Eksperimen_SML_lifani"
WORKSPACE = os.getenv("GITHUB_WORKSPACE", ".")

DAGSHUB_TOKEN = "62760274751f21f2f0211eee3fcc2900baad9594"
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

remote_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(remote_uri)
mlflow.set_experiment("Eksperimen_Prediksi_Status_Akademik")

def train_and_log_model():
    print("[INFO] Mengingesti matriks data bersih dari pipeline Kriteria 1...")
    # Menyambungkan rantai data dari Kriteria 1 ke Kriteria 2 
    data_path = os.path.join(WORKSPACE, "data_bersih.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[GALAT] Eksekusi Kriteria 1 terlebih dahulu. File tidak ada: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # \mathbf{X} \in \mathbb{R}^{B \times F}, \mathbf{y} \in \{0, 1, 2\}^{B}
    target_col = "Target"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Eksekusi Splitting secara lokal pada pipeline
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    base_model = RandomForestClassifier(random_state=42)
    param_space = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'class_weight': ['balanced'] 
    }
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_space, cv=3, scoring='accuracy', n_jobs=-1)
    
    print(f"[INFO] Memulai siklus pelatihan dan transmisi telemetri ke: {remote_uri}")
    with mlflow.start_run(run_name="Eksperimen_RandomForest_V2"):
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test_scaled)
        
        final_accuracy = accuracy_score(y_test, predictions)
        final_f1 = f1_score(y_test, predictions, average='weighted')
        
        # Pencatatan Parameter dan Metrik (Manual Logging) 
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("akurasi_pengujian", final_accuracy)
        mlflow.log_metric("f1_score_bobot", final_f1)
        mlflow.sklearn.log_model(best_model, "model_klasifikasi_mahasiswa")
        
        # Perekaman Artefak Tambahan: Confusion Matrix 
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        # Pemetaan statis sesuai label encoding Kriteria 1 (0: Dropout, 1: Enrolled, 2: Graduate)
        labels = ['Dropout', 'Enrolled', 'Graduate']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
        plt.title('Matriks Evaluasi Prediksi Model')
        plt.ylabel('Status Aktual')
        plt.xlabel('Hasil Prediksi')
        
        graph_filename = os.path.join(WORKSPACE, "evaluasi_confusion_matrix.png")
        plt.savefig(graph_filename, bbox_inches='tight')
        mlflow.log_artifact(graph_filename)
        plt.close()

        # Ekspor arsitektur untuk Serving / Docker (Kriteria 4)
        local_model_dir = os.path.join(WORKSPACE, "local_model_dir")
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
        mlflow.sklearn.save_model(best_model, local_model_dir)
        
        print("\n[SUKSES] Siklus Pelatihan Selesai.")

if __name__ == "__main__":
    train_and_log_model()