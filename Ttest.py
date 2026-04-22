import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_rel, shapiro


pln_5m = np.array([
27.8, 27.6, 28.4, 28.6, 28.9, 30.0, 27.3, 28.9, 27.6, 28.8,
30.7, 31.5, 31.0, 31.3, 30.8, 31.5, 31.0, 31.4, 30.9,
30.9, 31.3, 31.4, 31.5, 30.8, 26.6, 26.7, 27.3, 29.0, 29.5, 28.3
])

mlx_5m = np.array([
27.66, 27.58, 28.27, 28.54, 28.74, 29.95, 27.21, 28.83, 27.57, 28.70,
30.64, 31.26, 30.93, 31.13, 30.73, 31.34, 30.91, 31.17, 30.80,
30.77, 31.16, 31.17, 31.29, 30.73, 26.58, 26.68, 27.20, 28.90, 29.40, 28.22
])

pln_4m = np.array([
27.6, 28.2, 28.8, 28.0, 28.9, 29.3, 28.2, 28.8, 28.1,
29.8, 30.7, 30.9, 30.4, 31.5, 30.5, 31.4, 30.9, 31.6, 30.9,
31.1, 29.5, 31.6, 29.6, 31.3, 27.1, 27.5, 27.1, 29.6, 28.9, 28.7
])

mlx_4m = np.array([
27.68, 28.24, 28.66, 28.05, 28.87, 29.36, 28.14, 28.93, 28.03,
29.95, 30.76, 30.71, 30.53, 31.58, 30.46, 31.47, 30.86, 31.66, 30.85,
31.02, 29.63, 31.47, 29.74, 31.36, 27.05, 27.54, 27.03, 29.66, 28.84, 28.76
])

# ======================
# FUNGSI ANALISIS
# ======================

def analisis(mlx, pln, label):
    print(f"\n===== HASIL ANALISIS {label} =====")

    # RMSE
    rmse = np.sqrt(mean_squared_error(pln, mlx))

    # R2
    r2 = r2_score(pln, mlx)

    # Paired T-Test
    t_stat, p_val = ttest_rel(pln, mlx)

    # Normalitas (selisih)
    diff = pln - mlx
    shapiro_stat, shapiro_p = shapiro(diff)

    print(f"RMSE              : {rmse:.4f}")
    print(f"R2                : {r2:.4f}")
    print(f"P-value T-test    : {p_val:.4f}")
    print(f"P-value Normalitas: {shapiro_p:.4f}")

    # Interpretasi sederhana
    print("\nInterpretasi:")
    if p_val > 0.05:
        print("- Tidak ada perbedaan signifikan (baik)")
    else:
        print("- Ada perbedaan signifikan")

    if shapiro_p > 0.05:
        print("- Data berdistribusi normal (baik)")
    else:
        print("- Data tidak normal")


# ======================
# JALANKAN ANALISIS
# ======================

analisis(mlx_4m, pln_4m, "4 METER")
analisis(mlx_5m, pln_5m, "5 METER")
