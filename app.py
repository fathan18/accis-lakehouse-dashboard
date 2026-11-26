# ==============================================================================
# LAB 3 & LAB 4: INTERACTIVE LAKEHOUSE DASHBOARD + AI AUDITOR COPILOT (ACCIS)
# Membaca lapisan Gold via DuckDB + Visualisasi Plotly + Tanya Jawab Gemini AI
# ==============================================================================

import json
from datetime import datetime                       # Untuk memformat tanggal "Terakhir diperbarui"
import duckdb                                        # Query SQL langsung di atas file Parquet
import pandas as pd
import plotly.express as px                          # Grafik interaktif (bisa di-hover & zoom)
import streamlit as st                               # Kerangka aplikasi web Python
from pathlib import Path
from pipeline import run_pipeline                    # Panggil mesin pipeline dari Lab 1
from model_serving import run_batch_scoring          # Panggil mesin scoring dari Lab 2

# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(
    page_title="ACCIS Lakehouse · AI Audit Dashboard",
    page_icon="",
    layout="wide"                                    # Menggunakan lebar penuh layar monitor/TV
)

BASE   = Path(__file__).parent
GOLD   = BASE / "lake" / "gold"
WM     = BASE / "lake" / "bronze" / "watermark.json"

# Nama hari & bulan Indonesia agar tanggal tampil ramah dibaca
HARI  = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]

def format_tanggal_id(iso_text):
    """Ubah '2026-11-25T19:43:29+07:00' menjadi 'Rabu, 25 November 2026'."""
    try:
        dt = datetime.fromisoformat(iso_text)
        return f"{HARI[dt.weekday()]}, {dt.day} {BULAN[dt.month]} {dt.year}"
    except (TypeError, ValueError):
        return "-"                                       # Jika watermark belum ada / format tak dikenali

# Pastikan data Gold tersedia saat pertama kali dibuka (penting saat deploy ke Cloud!)
@st.cache_resource
def init_lakehouse():
    """Menjamin inisialisasi awal Lakehouse hanya berjalan sekali di Cloud."""
    if not (GOLD / "anomaly_flags.parquet").exists():
        run_pipeline()
        run_batch_scoring()
    return True

init_lakehouse()

# 2. FUNGSI MEMUAT DATA GOLD (DENGAN CACHE 60 DETIK AGAR RINGAN)
@st.cache_data(ttl=60)
def load_gold_data():
    """Membaca 3 tabel Gold menggunakan DuckDB."""
    monthly = duckdb.sql(f"SELECT * FROM '{(GOLD / 'monthly_account.parquet').as_posix()}'").df()
    assets  = duckdb.sql(f"SELECT * FROM '{(GOLD / 'asset_summary.parquet').as_posix()}'").df()
    flags   = duckdb.sql(f"SELECT * FROM '{(GOLD / 'anomaly_flags.parquet').as_posix()}'").df()
    return monthly, assets, flags

monthly_df, assets_df, flags_df = load_gold_data()

# Baca status pembaruan terakhir dari watermark.json
wm_info = json.loads(WM.read_text(encoding="utf-8")) if WM.exists() else {}

# 3. SIDEBAR KIRI: FILTER & KONTROL PIPELINE
with st.sidebar:
    st.header("⚙️ Kontrol Lakehouse")
    st.caption(f"Terakhir diperbarui: **{format_tanggal_id(wm_info.get('updated_at'))}**")
    st.caption(f"Watermark terakhir: `mjo_id = {wm_info.get('last_mjo_id', 0)}`")

    # Tombol simulasi penarikan data baru secara incremental + scoring otomatis
    if st.button("🔄 Tarik Data Baru (Incremental)", use_container_width=True, type="primary"):
        with st.spinner("Menjalankan Incremental Pipeline & Batch Scoring..."):
            run_pipeline()                           # Tarik +200 transaksi baru ke Bronze->Silver->Gold
            run_batch_scoring()                      # Nilai transaksi baru pakai model .joblib
            st.cache_data.clear()                    # Bersihkan cache agar angka langsung ter-update
        st.success("Data baru berhasil masuk & dinilai!")
        st.rerun()

    st.divider()
    # Filter tahun untuk grafik bulanan
    tahun_opsi = sorted(monthly_df["year"].unique().tolist(), reverse=True)
    tahun_pilih = st.selectbox("📅 Pilih Tahun Laporan", tahun_opsi)

    # Filter batas minimal skor anomali
    min_skor = st.slider("Batas Minimal Skor Anomali", 0.50, 0.85, 0.60, 0.01,
                         help="0.50 = normal, makin mendekati 1 = makin janggal")

    st.divider()
    # Input API Key Gemini gratis untuk fitur Tab 3 (AI Auditor Copilot)
    st.subheader("Pengaturan AI Copilot")
    api_key = st.text_input(
        "Gemini API Key (Gratis)",
        type="password",                             # Disembunyikan sebagai titik-titik
        help="Dapatkan gratis di aistudio.google.com/apikey"
    )

# 4. HEADER & KARTU KPI UTAMA
st.title("ACCIS Lakehouse & AI Audit Monitor")
st.markdown("Dashboard *read-only* yang membaca langsung lapisan **Gold** (Parquet + DuckDB) tanpa membebani server transaksi.")

df_yr = monthly_df[monthly_df["year"] == tahun_pilih]
df_fl = flags_df[flags_df["score"] >= min_skor]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Total Transaksi ({tahun_pilih})", f"Rp {df_yr['total'].sum()/1e9:,.2f} M")
c2.metric("Jumlah Jurnal Diproses", f"{int(df_yr['n_transaksi'].sum()):,} baris")
c3.metric("Jurnal Terindikasi Anomali", f"{len(df_fl)} temuan", delta="Perlu Audit", delta_color="inverse")
c4.metric("Total Nilai Buku Aset", f"Rp {assets_df['total_nilai_buku'].sum()/1e9:,.2f} M")

st.divider()

# 5. TIGA TAB UTAMA: TREN KEUANGAN | AUDIT ANOMALI | AI COPILOT
tab1, tab2, tab3 = st.tabs(["Ringkasan Keuangan & Aset", "Radar Anomali Jurnal", "AI Auditor Copilot"])

with tab1:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        # Grafik tren transaksi per bulan menggunakan Plotly
        tren = df_yr.groupby("month", as_index=False)["total"].sum()
        tren["juta_rp"] = tren["total"] / 1e6
        fig1 = px.bar(
            tren, x="month", y="juta_rp",
            title=f"Tren Total Transaksi Bulanan {tahun_pilih} (Juta Rp)",
            labels={"month": "Bulan", "juta_rp": "Nominal (Juta Rp)"},
            color_discrete_sequence=["#5b5bf0"]
        )
        fig1.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Grafik perbandingan nilai perolehan vs nilai buku aset per unit
        fig2 = px.bar(
            assets_df, x="unit", y=["total_perolehan", "total_nilai_buku"],
            barmode="group",
            title="Valuasi Aset Tetap per Unit Kerja",
            labels={"value": "Rupiah", "unit": "Unit Kerja", "variable": "Kategori"},
            color_discrete_sequence=["#06b6d4", "#10b981"]
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Daftar Jurnal Mencurigakan (Hasil Isolation Forest)")
    st.caption("Urutan dari skor paling janggal. Kolom `rasio_rata2` menunjukkan berapa kali lipat nominalnya dibanding kebiasaan akun tersebut.")

    # Tampilkan tabel interaktif (bisa diurutkan & dicari langsung di browser)
    st.dataframe(
        df_fl.style.format({"amount": "Rp {:,.0f}", "score": "{:.3f}", "rasio_rata2": "{:.1f}x"}),
        use_container_width=True,
        hide_index=True
    )

    # Tombol unduh hasil temuan ke CSV untuk bahan rapat audit
    csv_bytes = df_fl.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Daftar Anomali (.CSV)",
        data=csv_bytes,
        file_name=f"temuan_anomali_accis_{tahun_pilih}.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Tanya Jawab Bahasa Indonesia ke Data Gold ACCIS")
    st.info("AI hanya membaca ringkasan **lapisan Gold** (bukan data mentah/rahasia) agar jawaban akurat dan aman.")

    # Siapkan ringkasan konteks dari Gold untuk dikirim ke AI
    konteks_gold = (
        f"RINGKASAN TOP 10 ANOMALI JURNAL:\n{flags_df.head(10).to_string(index=False)}\n\n"
        f"RINGKASAN ASET PER UNIT:\n{assets_df.to_string(index=False)}\n\n"
        f"RINGKASAN TRANSAKSI TAHUN {tahun_pilih} PER AKUN:\n"
        f"{df_yr.groupby('acc_name', as_index=False)['total'].sum().to_string(index=False)}"
    )

    # Kotak pertanyaan pengguna
    tanya = st.text_area(
        "Tulis pertanyaan audit Anda:",
        placeholder="Contoh: Sebutkan 3 jurnal paling mencurigakan, siapa penginputnya, dan buat rekomendasi singkat untuk auditor."
    )

    if st.button("Tanya AI Auditor", type="primary"):
        if not api_key:
            st.warning("Masukkan **Gemini API Key** gratis Anda dulu di menu sidebar sebelah kiri.")
        elif not tanya.strip():
            st.warning("Silakan tulis pertanyaan terlebih dahulu.")
        else:
            try:
                from google import genai                 # SDK resmi Google Gemini
                client = genai.Client(api_key=api_key)
                prompt_final = (
                    "Anda adalah asisten auditor keuangan profesional untuk sistem ACCIS. "
                    "Jawablah dalam Bahasa Indonesia yang jelas, ringkas, dan berbasis angka dari data Gold berikut. "
                    "Jika data tidak tersedia, katakan dengan jujur.\n\n"
                    f"=== DATA GOLD ACCIS ===\n{konteks_gold}\n\n"
                    f"=== PERTANYAAN PENGGUNA ===\n{tanya}"
                )
                with st.spinner("AI Auditor sedang menganalisis data Gold..."):
                    resp = None
                    for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
                        try:
                            resp = client.models.generate_content(
                                model=model_name,
                                contents=prompt_final
                            )
                            break  # Berhasil, keluar dari loop dan tampilkan hasil
                        except Exception as err:
                            if any(code in str(err) for code in ["404", "503"]):
                                continue  # Coba model berikutnya jika tidak tersedia / sibuk
                            raise err

                if resp:
                    st.markdown(resp.text)
                else:
                    st.warning("Semua model AI sedang padat sementara. Silakan klik ulang dalam 10 detik.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi Gemini API: {e}")
