"""
Modul Fuzzy Mamdani - Tingkat Masalah Kulit
Input  : keminyakan, jerawat, iritasi (kemerahan+gatal), kekeringan  (skala 0-10)
Output : tingkat_masalah (0-100) -> ringan / sedang / berat
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------- 1. Variabel & fungsi keanggotaan ----------
x_in = np.arange(0, 10.01, 0.1)
x_out = np.arange(0, 100.01, 1)

def _buat_input(nama):
    v = ctrl.Antecedent(x_in, nama)
    v["rendah"] = fuzz.trapmf(v.universe, [0, 0, 2, 5])
    v["sedang"] = fuzz.trimf(v.universe, [2, 5, 8])
    v["tinggi"] = fuzz.trapmf(v.universe, [5, 8, 10, 10])
    return v

minyak = _buat_input("keminyakan")
jerawat = _buat_input("jerawat")
iritasi = _buat_input("iritasi")
kering = _buat_input("kekeringan")

masalah = ctrl.Consequent(x_out, "tingkat_masalah")
masalah["ringan"] = fuzz.trimf(masalah.universe, [0, 0, 40])
masalah["sedang"] = fuzz.trimf(masalah.universe, [20, 50, 80])
masalah["berat"] = fuzz.trimf(masalah.universe, [60, 100, 100])

# ---------- 2. Rule base (AND = min, komposisi = max) ----------
R = ctrl.Rule
rules = [
    R(iritasi["tinggi"], masalah["berat"]),
    R(jerawat["tinggi"] & minyak["tinggi"], masalah["berat"]),
    R(jerawat["tinggi"] & iritasi["sedang"], masalah["berat"]),
    R(kering["tinggi"] & iritasi["sedang"], masalah["berat"]),
    R(iritasi["sedang"], masalah["sedang"]),
    R(jerawat["sedang"], masalah["sedang"]),
    R(jerawat["tinggi"], masalah["sedang"]),
    R(minyak["tinggi"], masalah["sedang"]),
    R(kering["tinggi"], masalah["sedang"]),
    R(minyak["sedang"] & kering["sedang"], masalah["sedang"]),
    R(iritasi["rendah"] & jerawat["rendah"] & minyak["rendah"] & kering["rendah"], masalah["ringan"]),
    R(iritasi["rendah"] & jerawat["rendah"] & minyak["sedang"], masalah["ringan"]),
    R(iritasi["rendah"] & jerawat["rendah"] & kering["sedang"], masalah["ringan"]),
    R(iritasi["rendah"] & jerawat["rendah"] & minyak["rendah"], masalah["ringan"]),
]
_sistem = ctrl.ControlSystem(rules)

# ---------- 3. Inferensi + defuzzifikasi (centroid) ----------
def hitung(keminyakan, jerawat_, iritasi_, kekeringan):
    sim = ctrl.ControlSystemSimulation(_sistem)
    sim.input["keminyakan"] = float(np.clip(keminyakan, 0, 10))
    sim.input["jerawat"] = float(np.clip(jerawat_, 0, 10))
    sim.input["iritasi"] = float(np.clip(iritasi_, 0, 10))
    sim.input["kekeringan"] = float(np.clip(kekeringan, 0, 10))
    sim.compute()
    skor = float(sim.output["tingkat_masalah"])
    if skor < 35:
        label = "ringan"
    elif skor < 65:
        label = "sedang"
    else:
        label = "berat"
    return skor, label

# ---------- 4. Mapping rekomendasi (berdasarkan gejala dominan) ----------
REKOMENDASI = {
    "iritasi": "Produk menenangkan & perbaikan skin barrier (ceramide, panthenol, centella). Hindari dulu vitamin C, retinol, AHA/BHA.",
    "jerawat": "Produk anti-jerawat lembut (salicylic acid, niacinamide), non-comedogenic.",
    "keminyakan": "Pembersih gel & pelembap ringan oil-free, niacinamide untuk kontrol minyak.",
    "kekeringan": "Pelembap kaya (hyaluronic acid, ceramide, glycerin) dan pembersih tanpa sabun keras.",
    "normal": "Rutinitas dasar: pembersih lembut, pelembap, dan sunscreen.",
}

def rekomendasi(keminyakan, jerawat_, iritasi_, kekeringan):
    gejala = {"keminyakan": keminyakan, "jerawat": jerawat_, "iritasi": iritasi_, "kekeringan": kekeringan}
    dominan, nilai = max(gejala.items(), key=lambda kv: kv[1])
    if nilai < 3:
        dominan = "normal"
    return dominan, REKOMENDASI[dominan]

if __name__ == "__main__":
    # tes manual tanpa chatbot
    for kasus in [(2, 1, 7, 3), (8, 8, 2, 2), (1, 0, 0, 1), (3, 2, 9, 6), (2, 9, 1, 1)]:
        s, l = hitung(*kasus)
        print(kasus, "->", round(s, 1), l, "|", rekomendasi(*kasus)[0])

