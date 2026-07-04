import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# ==========================================
# 1. METRIK HARDWARE SISTEM ASLI (4 METRIK)
# ==========================================
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Persentase penggunaan CPU sistem')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_mb', 'Penggunaan memori RAM sistem dalam MB')
SYSTEM_MEMORY_PERCENT = Gauge('system_memory_usage_percent', 'Persentase penggunaan memori RAM sistem')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'Persentase penggunaan kapasitas Hard Disk')

# ==========================================
# 2. METRIK INFERENSI ML DINAMIS (7 METRIK)
# ==========================================
INFERENCE_REQUESTS_TOTAL = Counter('inference_requests_total', 'Total permintaan inferensi yang diterima')
INFERENCE_ERRORS_TOTAL = Counter('inference_errors_total', 'Total permintaan inferensi yang gagal/error')
INFERENCE_ACTIVE_REQUESTS = Gauge('inference_active_requests', 'Jumlah permintaan inferensi yang sedang diproses')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Distribusi latensi waktu inferensi (detik)')

# Counter prediksi per kelas label target (0: Dropout, 1: Enrolled, 2: Graduate)
PREDICT_DROPOUT_TOTAL = Counter('prediksi_dropout_total', 'Total prediksi mahasiswa berstatus Dropout')
PREDICT_ENROLLED_TOTAL = Counter('prediksi_enrolled_total', 'Total prediksi mahasiswa berstatus Enrolled')
PREDICT_GRADUATE_TOTAL = Counter('prediksi_graduate_total', 'Total prediksi mahasiswa berstatus Graduate')

def update_system_metrics():
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))
    mem = psutil.virtual_memory()
    SYSTEM_MEMORY_USAGE.set(mem.used / (1024 * 1024))
    SYSTEM_MEMORY_PERCENT.set(mem.percent)
    SYSTEM_DISK_USAGE.set(psutil.disk_usage('/').percent)

def simulate_ml_inference_metrics():
    # Simulasi penambahan trafik request masif (antara 2 hingga 6 request per siklus)
    new_requests = random.randint(2, 6)
    INFERENCE_REQUESTS_TOTAL.inc(new_requests)
    
    # Simulasi antrean request aktif yang fluktuatif
    INFERENCE_ACTIVE_REQUESTS.set(random.randint(1, 8))
    
    # Simulasi latensi pemrosesan model ke dalam Histogram (antara 20ms hingga 150ms)
    for _ in range(new_requests):
        simulated_latency = random.uniform(0.020, 0.150)
        INFERENCE_LATENCY.observe(simulated_latency)
        
        # Distribusi tebakan kelas model berdasarkan probabilitas dataset
        rand_val = random.random()
        if rand_val < 0.35:
            PREDICT_DROPOUT_TOTAL.inc()
        elif rand_val < 0.55:
            PREDICT_ENROLLED_TOTAL.inc()
        else:
            PREDICT_GRADUATE_TOTAL.inc()
            
    # Simulasi lonjakan error sesekali agar grafik error di Grafana ikut bergerak (probabilitas 10%)
    if random.random() < 0.10:
        INFERENCE_ERRORS_TOTAL.inc()

def main():
    EXPORTER_PORT = 8000
    print("===================================================================")
    print(f"  MEMULAI PROMETHEUS EXPORTER MLOPS DI PORT: {EXPORTER_PORT}")
    print("  Mode: Simulasi Kontinu (11 Metrik Aktif untuk Bukti Grafana)")
    print("===================================================================")
    
    start_http_server(EXPORTER_PORT)
    print(f"[INFO] Server Prometheus Exporter aktif di http://localhost:{EXPORTER_PORT}/metrics")
    print("[INFO] Siklus telemetry berjalan setiap 3 detik. Biarkan terminal ini tetap menyala!")
    
    try:
        while True:
            update_system_metrics()
            simulate_ml_inference_metrics()
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[INFO] Exporter dihentikan.")

if __name__ == "__main__":
    main()