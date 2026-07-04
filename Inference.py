import time
import json
import random
import requests

# Konfigurasi Endpoint Server Inferensi Model (Docker Container di Port 5001)
INFERENCE_URL = "http://localhost:5001/invocations"
CUSTOM_URL_FALLBACK = "http://localhost:5001/predict"

# Daftar 35 Fitur Input sesuai spesifikasi dataset.csv (Tanpa kolom Target)
FEATURE_COLUMNS = [
    'Marital status', 'Application mode', 'Application order', 'Course', 
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality', 
    "Mother's qualification", "Father's qualification", "Mother's occupation", 
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 
    'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 
    'Unemployment rate', 'Inflation rate', 'GDP'
]

def generate_dummy_student_data():
    """
    Menghasilkan satu vektor data mahasiswa sintetis X in R^(1x35) dengan variasi rasional.
    """
    # Profil representatif untuk memicu variasi prediksi kelas (0: Dropout, 1: Enrolled, 2: Graduate)
    profile_type = random.choice(["high_performer", "at_risk", "average"])
    
    if profile_type == "high_performer":
        sem1_approved, sem1_grade = random.randint(5, 7), round(random.uniform(13.5, 18.0), 2)
        sem2_approved, sem2_grade = random.randint(5, 7), round(random.uniform(14.0, 18.5), 2)
        debtor, tuition_up_to_date = 0, 1
    elif profile_type == "at_risk":
        sem1_approved, sem1_grade = random.randint(0, 2), round(random.uniform(0.0, 10.0), 2)
        sem2_approved, sem2_grade = random.randint(0, 1), round(random.uniform(0.0, 9.5), 2)
        debtor, tuition_up_to_date = random.choice([0, 1]), 0
    else:
        sem1_approved, sem1_grade = random.randint(3, 5), round(random.uniform(11.0, 13.5), 2)
        sem2_approved, sem2_grade = random.randint(3, 5), round(random.uniform(11.0, 13.5), 2)
        debtor, tuition_up_to_date = 0, 1

    data_record = [
        1, 1, 1, random.choice([2, 5, 11, 17, 9500]), 1, 1, 1, 13, 10, 6, 10,
        random.choice([0, 1]), 0, debtor, tuition_up_to_date, random.choice([0, 1]), 
        random.choice([0, 1]), random.randint(18, 30), 0, 0, 6, 6, sem1_approved, 
        sem1_grade, 0, 0, 6, 6, sem2_approved, sem2_grade, 0, 
        round(random.uniform(7.0, 16.0), 1), round(random.uniform(-0.8, 3.5), 2), round(random.uniform(-4.0, 3.5), 2)
    ]
    
    return data_record

def send_inference_request(url, payload_format="mlflow"):
    record = generate_dummy_student_data()
    
    if payload_format == "mlflow":
        # Format standar MLflow Model Serving (dataframe_split)
        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLUMNS,
                "data": [record]
            }
        }
    else:
        # Format dictionary records kustom untuk FastAPI/Flask
        payload = dict(zip(FEATURE_COLUMNS, record))
        
    headers = {"Content-Type": "application/json"}
    
    try:
        start_time = time.time()
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=5)
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SUKSES] Status: 200 | Latensi: {latency_ms:.1f}ms | Prediksi Kelas: {result}")
            return True
        else:
            print(f"[GALAT HTTP {response.status_code}] Respons: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"[KONEKSI GAGAL] Tidak dapat terhubung ke {url}. Pastikan kontainer Docker server aktif!")
        return False
    except Exception as e:
        print(f"[EXCEPT] Galat tidak terduga: {str(e)}")
        return False

def run_traffic_generator(total_requests=50, delay_seconds=1.5):
    print("===================================================================")
    print("  MEMULAI GENERATOR TRAFIK INFERENSI UNTUK PEMANTAUAN GRAFANA")
    print("===================================================================")
    print(f"Target URL      : {INFERENCE_URL} (Fallback: {CUSTOM_URL_FALLBACK})")
    print(f"Total Permintaan: {total_requests} Request")
    print("===================================================================\n")
    
    active_url = INFERENCE_URL
    format_type = "mlflow"
    
    print("[INFO] Menguji koneksi awal ke endpoint utama...")
    if not send_inference_request(active_url, format_type):
        print(f"[INFO] Mencoba endpoint fallback: {CUSTOM_URL_FALLBACK} dengan format kustom...")
        active_url = CUSTOM_URL_FALLBACK
        format_type = "custom"
        if not send_inference_request(active_url, format_type):
            print("[FATAL] Kedua endpoint tidak merespons! Hentikan eksekusi. Pastikan Docker port 5001 menyala.")
            return

    print("\n[INFO] Koneksi terverifikasi. Memulai transmisi trafik beruntun...")
    success_count = 0
    
    for i in range(1, total_requests + 1):
        print(f"Request [{i:02d}/{total_requests:02d}] -> ", end="")
        is_success = send_inference_request(active_url, format_type)
        if is_success:
            success_count += 1
        
        # Jeda waktu acak agar grafik trafik di Prometheus terlihat natural
        time.sleep(random.uniform(0.5, delay_seconds))
        
    print("\n===================================================================")
    print(f"  EKSEKUSI SELESAI: {success_count}/{total_requests} Permintaan Berhasil Dilayani.")
    print("  SEGERA BUKA DASBOR GRAFANA ANDA DAN AMBIL SCREENSHOT GRAFIKNYA!")
    print("===================================================================")

if __name__ == "__main__":
    # Mengirimkan 50 request secara otomatis saat skrip dijalankan
    run_traffic_generator(total_requests=50, delay_seconds=1.0)