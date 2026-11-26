# ==============================================================================
# LAB 1: MODULAR LAKEHOUSE PIPELINE + INCREMENTAL WATERMARK (ACCIS)
# Menjalankan: Simulasi Sumber -> Bronze (Incremental) -> Silver (Gate) -> Gold
# ==============================================================================

import os                       # Operasi sistem file & atomic rename (os.replace)
import json                     # Membaca & menulis file metadata watermark.json
import logging                  # Mencatat riwayat proses ke file pipeline.log
from datetime import datetime, timezone, timedelta   # Stempel waktu ekstraksi (zona WIB)
from pathlib import Path        # Mengelola alamat folder agar aman di Windows & Mac
import numpy as np              # Generator angka acak untuk data sintetis ACCIS
import pandas as pd             # Pengolahan tabel data (DataFrame)
import duckdb                   # Mesin SQL lokal di atas file Parquet
import uuid

# 1. KONFIGURASI FOLDER & LOGGING
BASE = Path(__file__).parent                       # Lokasi folder project saat ini
BRONZE = BASE / "lake" / "bronze"                  # Folder lapisan Bronze
SILVER = BASE / "lake" / "silver"                  # Folder lapisan Silver
GOLD   = BASE / "lake" / "gold"                    # Folder lapisan Gold
WATERMARK_FILE = BRONZE / "watermark.json"         # File pembatas buku (incremental)

# Pastikan semua folder tersedia; tidak error jika sudah ada
for folder in [BRONZE, SILVER, GOLD, BASE / "models"]:
    folder.mkdir(parents=True, exist_ok=True)

# Atur pencatatan log ke file 'pipeline.log' sekaligus tampil di layar terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(BASE / "pipeline.log", encoding="utf-8"),
              logging.StreamHandler()]
)

def atomic_to_parquet(df: pd.DataFrame, target_path: Path):
    """Menyimpan DataFrame ke Parquet secara ATOMIK dengan nama file sementara unik."""
    tmp_path = target_path.with_name(f".{target_path.stem}_{uuid.uuid4().hex}.tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, target_path)

def load_watermark() -> dict:
    """Membaca ID jurnal terakhir yang pernah ditarik (seperti pembatas buku)."""
    if WATERMARK_FILE.exists():
        return json.loads(WATERMARK_FILE.read_text(encoding="utf-8"))
    return {"last_mjo_id": 0, "run_count": 0}            # Nilai awal jika baru pertama jalan

def save_watermark(wm: dict):
    """Menyimpan penanda ID terakhir yang berhasil masuk ke Bronze."""
    WATERMARK_FILE.write_text(json.dumps(wm, indent=2), encoding="utf-8")

# 2. SIMULASI SUMBER DATABASE ACCIS (MANDIRI / CREATE NEW PROJECT)
def generate_accis_batch(start_id: int, n_rows: int, seed: int):
    """Membangkitkan batch data baru menyerupai skema asasta_accis.sql."""
    rng = np.random.default_rng(seed)                    # Seed berubah per batch agar data baru unik

    # Master akun perkiraan (meniru masteraccount_perkiraan)
    acc = pd.DataFrame({
        "acc_id": range(1, 9),
        "acc_code": ["1101", "1201", "2101", "4101", "5101", "5102", "5201", "5301"],
        "acc_name": ["Kas & Bank", "Persediaan", "Hutang Usaha", "Pendapatan Air",
                     "Beban Gaji", "Beban Listrik", "Beban Pemeliharaan", "Beban Penyusutan"]
    })

    # Tanggal transaksi sepanjang 2024-2025
    ids = np.arange(start_id + 1, start_id + n_rows + 1)
    tgl = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n_rows), unit="D")

    # Header jurnal (meniru mjournal)
    mj = pd.DataFrame({
        "mjo_id": ids,
        "mjo_code": [f"JV-{i:05d}" for i in ids],
        "mjo_date": tgl.strftime("%Y-%m-%d"),
        "mjo_desc": rng.choice(["Bayar listrik", "Bayar gaji", "Beli ATK",
                                "Perbaikan pipa", "Terima pembayaran", "Setor kas"], n_rows),
        "mjo_isposted": rng.choice([0, 1], n_rows, p=[0.04, 0.96]),
        "mjo_isactive": 1,
        "of_id": rng.integers(1, 6, n_rows),
        "input_by": rng.choice(["siti", "budi", "andi", "rina"], n_rows)
    })

    # Detail jurnal (meniru djournal)
    dj = pd.DataFrame({
        "djo_id": ids,
        "mjo_id": ids,
        "acc_id": rng.integers(1, 9, n_rows),
        "djo_debit": np.round(rng.lognormal(14.5, 0.6, n_rows), 0),
        "djo_credit": 0.0
    })

    # Sisipkan ~1% anomali bisnis (nominal melonjak 60x-90x dari normal)
    n_anom = max(2, int(n_rows * 0.01))
    bad_idx = rng.choice(n_rows, n_anom, replace=False)
    dj.loc[bad_idx, "djo_debit"] *= rng.uniform(60, 90, n_anom)
    dj["djo_debit"] = np.round(dj["djo_debit"], 0)

    # Sisipkan sedikit kotoran data agar Quality Gate bekerja nyata
    if n_rows >= 50:
        idx_tgl = rng.choice(n_rows, 10, replace=False)
        mj.loc[idx_tgl, "mjo_date"] = pd.to_datetime(
            mj.loc[idx_tgl, "mjo_date"]
        ).dt.strftime("%d/%m/%Y")                        # Format tanggal campur
        mj.loc[rng.choice(n_rows, 5, replace=False), "input_by"] = None   # Missing value
        mj = pd.concat([mj, mj.sample(5, random_state=seed)], ignore_index=True)  # Duplikat

    # Aset tetap (meniru aktiva_data) - 400 baris
    m = 400
    rng_ak = np.random.default_rng(100)
    harga = np.round(rng_ak.lognormal(16, 0.8, m), 0)
    masa = rng_ak.choice([48, 96, 192, 240], m)
    pakai = np.minimum(rng_ak.integers(1, 260, m), masa)
    ak = pd.DataFrame({
        "kode_asset": [f"AST-{i:04d}" for i in range(m)],
        "unit": rng_ak.choice(["U01 - Pusat", "U02 - Timur", "U03 - Barat"], m),
        "nama_asset": rng_ak.choice(["Pompa", "Pipa transmisi", "Meter air", "Kendaraan", "Komputer"], m),
        "harga_total": harga,
        "nilai_buku": np.round(harga * (1 - pakai / masa), 0)
    })
    ak.loc[rng_ak.choice(m, 6, replace=False), "nilai_buku"] = -1000  # Kotoran: aset minus

    return acc, mj, dj, ak

# 3. FUNGSI UTAMA PIPELINE: BRONZE -> SILVER -> GOLD
def run_pipeline(batch_size: int = 3000):
    wm = load_watermark()
    start_id = wm["last_mjo_id"]
    run_no = wm["run_count"] + 1
    logging.info(f"=== Mulai Pipeline Run #{run_no} (Watermark terakhir mjo_id={start_id}) ===")

    # Jika sudah pernah jalan, simulasi batch tambahan kecil (misal 200 transaksi baru)
    n_new = batch_size if start_id == 0 else 200
    acc, mj_new, dj_new, ak = generate_accis_batch(start_id, n_new, seed=42 + run_no)

    # --- TAHAP 1: INGEST INCREMENTAL KE BRONZE ---
    now_ts = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")   # WIB (UTC+7), server Cloud default-nya UTC
    for df_part, name in [(mj_new, "mjournal"), (dj_new, "djournal")]:
        df_part["_source_table"] = name
        df_part["_ingested_at"] = now_ts
        p_file = BRONZE / f"{name}.parquet"
        if p_file.exists():
            # Gabungkan data lama di Bronze dengan batch incremental yang baru datang
            old_df = pd.read_parquet(p_file)
            combined = pd.concat([old_df, df_part], ignore_index=True)
            atomic_to_parquet(combined, p_file)
        else:
            atomic_to_parquet(df_part, p_file)

    # Tabel master & aktiva disimpan ulang (snapshot)
    atomic_to_parquet(acc, BRONZE / "masteraccount_perkiraan.parquet")
    atomic_to_parquet(ak, BRONZE / "aktiva_data.parquet")
    logging.info(f"[BRONZE] Berhasil ingest {n_new} transaksi baru (mjo_id {start_id+1}..{start_id+n_new}).")

    # --- TAHAP 2: BRONZE -> SILVER + QUALITY GATE ---
    mj = pd.read_parquet(BRONZE / "mjournal.parquet").drop_duplicates("mjo_id")
    d1 = pd.to_datetime(mj["mjo_date"], format="%Y-%m-%d", errors="coerce")
    d2 = pd.to_datetime(mj["mjo_date"], format="%d/%m/%Y", errors="coerce")
    mj["mjo_date"] = d1.fillna(d2)                       # Seragamkan format tanggal
    mj["input_by"] = mj["input_by"].fillna("unknown")    # Tangani missing value

    dj = pd.read_parquet(BRONZE / "djournal.parquet").drop_duplicates("djo_id")
    acc_b = pd.read_parquet(BRONZE / "masteraccount_perkiraan.parquet")

    # Gabungkan detail + header + akun lewat ID master
    jr = (dj.drop(columns=["_source_table", "_ingested_at"])
            .merge(mj.drop(columns=["_source_table", "_ingested_at"]), on="mjo_id")
            .merge(acc_b, on="acc_id"))
    jr["amount"] = jr["djo_debit"] + jr["djo_credit"]
    jr["year"]   = jr["mjo_date"].dt.year
    jr["month"]  = jr["mjo_date"].dt.month

    # Karantina aset yang nilai bukunya minus
    ak_b = pd.read_parquet(BRONZE / "aktiva_data.parquet")
    karantina = ak_b[ak_b["nilai_buku"] < 0]
    ak_clean  = ak_b[ak_b["nilai_buku"] >= 0]
    atomic_to_parquet(karantina, SILVER / "aset_karantina.parquet")

    # Quality Gate (wajib lolos semua)
    checks = {
        "djo_id_unik": bool(jr["djo_id"].is_unique),
        "tanggal_valid": bool(jr["mjo_date"].notna().all()),
        "amount_positif": bool((jr["amount"] >= 0).all()),
        "kelengkapan_input_by_95": bool((jr["input_by"] != "unknown").mean() >= 0.95),
        "nilai_buku_positif": bool((ak_clean["nilai_buku"] >= 0).all()),
    }
    if not all(checks.values()):
        logging.error(f"[QUALITY GATE GAGAL] Detail: {checks}")
        raise RuntimeError("Quality Gate gagal! Proses dihentikan sebelum menyentuh Silver.")

    atomic_to_parquet(jr, SILVER / "journal_line.parquet")
    atomic_to_parquet(ak_clean, SILVER / "aktiva.parquet")
    logging.info(f"[SILVER] Lolos Quality Gate: {len(jr)} jurnal bersih, {len(ak_clean)} aset ({len(karantina)} karantina).")

    # --- TAHAP 3: SILVER -> GOLD (AGREGASI SIAP DASHBOARD & AI) ---
    g1 = jr.groupby(["year", "month", "acc_code", "acc_name"], as_index=False).agg(
        total=("amount", "sum"),
        n_transaksi=("djo_id", "count")
    )
    g2 = ak_clean.groupby("unit", as_index=False).agg(
        jumlah_aset=("kode_asset", "count"),
        total_perolehan=("harga_total", "sum"),
        total_nilai_buku=("nilai_buku", "sum")
    )
    atomic_to_parquet(g1, GOLD / "monthly_account.parquet")
    atomic_to_parquet(g2, GOLD / "asset_summary.parquet")

    # Perbarui pembatas buku (watermark) setelah semua tahap sukses
    save_watermark({"last_mjo_id": int(start_id + n_new), "run_count": run_no, "updated_at": now_ts})
    logging.info("[GOLD] Tabel ringkasan terbit secara atomik. Pipeline SELESAI.\n")

if __name__ == "__main__":
    run_pipeline()
