# ==============================================================================
# LAB 2: PEMISAHAN TRAINING & BATCH SCORING (MLOPS RINGAN)
# Menyimpan model ke .joblib & menerbitkan anomaly_flags.parquet secara atomik
# ==============================================================================

import os
import sys
import joblib                                        # Pustaka untuk menyimpan/memuat model ML
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
from sklearn.ensemble import IsolationForest         # Algoritma deteksi anomali dari Modul 5

BASE       = Path(__file__).parent
SILVER     = BASE / "lake" / "silver"
GOLD       = BASE / "lake" / "gold"
MODEL_PATH = BASE / "models" / "iso_accis_v1.joblib" # Lokasi file model terlatih

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan fitur numerik yang konsisten untuk Training maupun Scoring."""
    out = df.copy()
    out["log_amt"] = np.log1p(out["amount"])         # Logaritma meredam selisih nominal besar
    out["dow"]     = out["mjo_date"].dt.dayofweek    # Hari transaksi (0=Senin .. 6=Minggu)
    return out[["log_amt", "acc_id", "dow"]]

def train_and_save_model():
    """TAHAP TRAINING: dijalankan berkala (misal mingguan/bulanan)."""
    jr = pd.read_parquet(SILVER / "journal_line.parquet")
    X  = prepare_features(jr)

    # Latih Isolation Forest pada data Silver
    iso = IsolationForest(
        n_estimators=200,                            # 200 pohon isolasi
        contamination=0.015,                         # Estimasi ~1.5% transaksi janggal
        random_state=42                              # Agar hasil konsisten
    ).fit(X)

    # Bekukan model ke dalam file .joblib agar tidak perlu dilatih ulang tiap hari
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, MODEL_PATH)
    print(f"✅ [TRAIN] Model berhasil dilatih ({len(jr)} baris) & disimpan di: {MODEL_PATH.name}")

def run_batch_scoring():
    """TAHAP INFERENCE / SCORING: dijalankan cepat tiap siklus pipeline."""
    # Jika file model belum ada, latih otomatis terlebih dahulu
    if not MODEL_PATH.exists():
        print("ℹ️ Model belum ditemukan. Menjalankan training awal...")
        train_and_save_model()

    # Muat model yang sudah jadi (hanya butuh milidetik!)
    iso = joblib.load(MODEL_PATH)
    jr  = pd.read_parquet(SILVER / "journal_line.parquet")
    X   = prepare_features(jr)

    # Hitung skor anomali (makin besar = makin janggal) & status prediksi
    jr["score"]    = -iso.score_samples(X)
    jr["is_flag"]  = (iso.predict(X) == -1).astype(int)

    # Ambil transaksi yang ditandai janggal, urutkan dari skor tertinggi
    flagged = jr[jr["is_flag"] == 1].sort_values("score", ascending=False).copy()
    flagged["mjo_date"] = flagged["mjo_date"].dt.strftime("%Y-%m-%d")

    # Hitung rasio terhadap rata-rata akun agar auditor cepat paham ALASAN kejanggalannya
    avg_acc = jr.groupby("acc_id")["amount"].mean().to_dict()
    flagged["rasio_rata2"] = (flagged["amount"] / flagged["acc_id"].map(avg_acc)).round(1)

    # Tulis ke Gold secara ATOMIK (tulis .tmp dulu, lalu tukar nama instan)
    cols = ["djo_id", "mjo_code", "mjo_date", "acc_code", "acc_name",
            "amount", "rasio_rata2", "score", "input_by", "mjo_desc"]
    target = GOLD / "anomaly_flags.parquet"
    tmp    = target.with_name(f".{target.stem}_{uuid.uuid4().hex}.tmp.parquet")
    flagged[cols].to_parquet(tmp, index=False)
    os.replace(tmp, target)

    print(f"✅ [SCORING] {len(flagged)} jurnal mencurigakan diterbitkan ke Gold secara atomik.")

if __name__ == "__main__":
    # Bisa dipanggil dengan: python model_serving.py train  ATAU  python model_serving.py score
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "train":
        train_and_save_model()
    elif mode == "score":
        run_batch_scoring()
    else:
        train_and_save_model()
        run_batch_scoring()
