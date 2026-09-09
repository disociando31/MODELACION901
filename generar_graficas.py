"""Genera las graficas de soporte (histogramas y dispersion) para el informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generador_numeros_aleatorios import (
    GeneradorCongruencialLineal, GeneradorCuadradosMedios
)

SEMILLA = 12345
N = 1000

lcg = GeneradorCongruencialLineal(semilla=SEMILLA)
datos_lcg = lcg.generar_secuencia(N, 0.0, 1.0)

semilla_ms = 1235  # impar, derivada de la semilla general
ms = GeneradorCuadradosMedios(semilla=semilla_ms, n_digitos=4)
datos_ms = ms.generar_secuencia(N, 0.0, 1.0)

# ---------- Histogramas (uniformidad) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].hist(datos_lcg, bins=20, color="#2E7D5B", edgecolor="white")
axes[0].set_title("LCG — Histograma (n=1000)")
axes[0].set_xlabel("Valor generado [0,1)")
axes[0].set_ylabel("Frecuencia")
axes[0].axhline(N / 20, color="black", linestyle="--", linewidth=1, label="Frec. esperada")
axes[0].legend(fontsize=8)

axes[1].hist(datos_ms, bins=20, color="#B0413E", edgecolor="white")
axes[1].set_title("Cuadrados Medios — Histograma (n=1000)")
axes[1].set_xlabel("Valor generado [0,1)")
axes[1].set_ylabel("Frecuencia")
axes[1].axhline(N / 20, color="black", linestyle="--", linewidth=1, label="Frec. esperada")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("histogramas.png", dpi=150)
plt.close()

# ---------- Dispersion X(n) vs X(n+1) (independencia) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

axes[0].scatter(datos_lcg[:-1], datos_lcg[1:], s=6, alpha=0.5, color="#2E7D5B")
axes[0].set_title("LCG — X(n) vs X(n+1)")
axes[0].set_xlabel("X(n)")
axes[0].set_ylabel("X(n+1)")

axes[1].scatter(datos_ms[:-1], datos_ms[1:], s=6, alpha=0.5, color="#B0413E")
axes[1].set_title("Cuadrados Medios — X(n) vs X(n+1)")
axes[1].set_xlabel("X(n)")
axes[1].set_ylabel("X(n+1)")

plt.tight_layout()
plt.savefig("dispersión_independencia.png", dpi=150)
plt.close()

print("Graficas generadas: histogramas.png, dispersión_independencia.png")
