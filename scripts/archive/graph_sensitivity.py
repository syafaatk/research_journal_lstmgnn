# -*- coding: utf-8 -*-
"""
Sensitivitas graf terhadap sumber koordinat (item A1):
Bandingkan graf k=3 threshold 95.82 km yang dibangun dari:
  (a) mean lokasi pelanggan (dipakai eksperimen)
  (b) koordinat ibu kota kabupaten/kota (referensi publik, aproksimasi)
"""
import json
import numpy as np

REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Musi Rawas Utara", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

# Koordinat eksperimen (mean lokasi pelanggan)
exp = json.load(open(r"E:\Download\Jurnal\skrip\coords_experiment.json"))

# Koordinat ibu kota (aproksimasi dari pengetahuan umum; PERLU verifikasi penulis)
CAP = {
    "Banyuasin": (-2.8833, 104.7500),          # Pangkalan Balai
    "Empat Lawang": (-3.5667, 103.0167),       # Tebing Tinggi
    "Lahat": (-3.7864, 103.5428),
    "Lubuk Linggau": (-3.2931, 102.8550),
    "Muara Enim": (-3.6516, 103.7708),
    "Musi Banyuasin": (-2.8833, 103.8333),     # Sekayu
    "Musi Rawas": (-3.1333, 103.2000),         # Muara Beliti
    "Musi Rawas Utara": (-3.0167, 103.0167),   # Muara Rupit
    "Ogan Ilir": (-3.2167, 104.6667),          # Indralaya
    "Ogan Komering Ilir": (-3.3833, 104.8333), # Kayuagung
    "Ogan Komering Ulu": (-4.1167, 104.1667),  # Baturaja
    "Ogan Komering Ulu Selatan": (-4.5333, 104.0833),  # Muaradua
    "Ogan Komering Ulu Timur": (-4.3000, 104.3000),    # Martapura
    "Pagar Alam": (-4.0250, 103.2500),
    "Palembang": (-2.9911, 104.7567),
    "Penukal Abab Lematang Ilir": (-3.2833, 104.0000), # Talang Ubi
    "Prabumulih": (-3.4333, 104.2333),
}

def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))

def build_graph(coords, k=3, threshold=95.82):
    """k-NN undirected tanpa self-loop, threshold jarak."""
    n = len(REGIONS)
    edges = set()
    for i in range(n):
        d = [(haversine(coords[REGIONS[i]], coords[REGIONS[j]]), j) for j in range(n) if j != i]
        d.sort()
        for dist, j in d[:k]:
            if dist <= threshold:
                edges.add(tuple(sorted((i, j))))
    return edges

# Jarak mean-pelanggan vs ibu kota per region
print("=== Deviasi koordinat eksperimen vs ibu kota (km) ===")
devs = {}
for r in REGIONS:
    d = haversine((exp[r]["lat"], exp[r]["lon"]), CAP[r])
    devs[r] = d
    print(f"  {r:<28} {d:>8.1f}")
dev = np.array(list(devs.values()))
print(f"  median {np.median(dev):.1f}, max {dev.max():.1f} km")

# Graf eksperimen vs graf ibu kota
exp_t = {r: (exp[r]["lat"], exp[r]["lon"]) for r in REGIONS}
g_exp = build_graph(exp_t)
g_cap = build_graph(CAP)
print(f"\n=== Perbandingan graf (k=3, threshold 95.82 km) ===")
print(f"  edge eksperimen: {len(g_exp)}")
print(f"  edge ibu kota:   {len(g_cap)}")
print(f"  edge sama:       {len(g_exp & g_cap)}")
print(f"  edge hanya eksperimen: {len(g_exp - g_cap)}")
print(f"  edge hanya ibu kota:   {len(g_cap - g_exp)}")
for e in sorted(g_exp - g_cap):
    print(f"    - {REGIONS[e[0]]} -- {REGIONS[e[1]]}")
for e in sorted(g_cap - g_exp):
    print(f"    + {REGIONS[e[0]]} -- {REGIONS[e[1]]}")

# Derajat
def degrees(g):
    deg = {r: 0 for r in REGIONS}
    for i, j in g:
        deg[REGIONS[i]] += 1
        deg[REGIONS[j]] += 1
    return deg
d_exp = degrees(g_exp)
d_cap = degrees(g_cap)
print(f"\n  derajat eksperimen: min {min(d_exp.values())}, max {max(d_exp.values())}, rata {np.mean(list(d_exp.values())):.2f}")
print(f"  derajat ibu kota:   min {min(d_cap.values())}, max {max(d_cap.values())}, rata {np.mean(list(d_cap.values())):.2f}")