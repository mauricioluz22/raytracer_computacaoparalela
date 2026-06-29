
import os
import time
import subprocess
import matplotlib.pyplot as plt

# ==============================
# 🔹 CONFIGURAÇÕES
# ==============================

# Configurações de Carga
THREADS = [1, 2, 3, 4, 6, 8, 12]
LARGURA_BASE = 1200
LARGURA_BASE_WEAK = 400
OUTPUT_DIR = "resultados_ppm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXECUTAVEL = "./build/main"


# ==============================
# 🔹 FUNÇÃO DE MEDIÇÃO
# ==============================

def medir_tempo(width, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    start = time.time()
    with open(f"{OUTPUT_DIR}/img_t{threads}_w{width}.ppm", "w") as f:
        subprocess.run(
            [EXECUTAVEL, str(width)],
            env=env,
            stdout=f,
            stderr=subprocess.DEVNULL
        )
    return time.time() - start


# ==============================
# 🔹 STRONG SCALING
# ==============================

print(f"\n{'='*75}")
print("STRONG SCALING (problema fixo)")
print(f"{'THREADS':<10} | {'LARGURA':<10} | {'TEMPO (s)':<15} | {'SPEEDUP':<10} | {'EFICIÊNCIA':<10}")
print(f"{'='*75}")

tempos_strong = []
t1 = None

for t in THREADS:
    tempo = medir_tempo(LARGURA_BASE, t)
    tempos_strong.append(tempo)

    if t1 is None:
        t1 = tempo

    speedup = t1 / tempo
    eficiencia = speedup / t

    print(f"{t:<10} | {LARGURA_BASE:<10} | {tempo:<15.4f} | {speedup:<10.2f} | {eficiencia:<10.2f}")

print(f"{'='*75}\n")


# ==============================
# 🔹 WEAK SCALING
# ==============================

print(f"\n{'='*75}")
print("WEAK SCALING (problema cresce com threads)")
print(f"{'THREADS':<10} | {'LARGURA':<10} | {'TEMPO (s)':<15}")
print(f"{'='*75}")

tempos_weak = []

for t in THREADS:
    largura = LARGURA_BASE_WEAK * t
    tempo = medir_tempo(largura, t)
    tempos_weak.append(tempo)

    print(f"{t:<10} | {largura:<10} | {tempo:<15.4f}")

print(f"{'='*75}\n")


# ==============================
# 🔹 GRÁFICO DE SPEEDUP
# ==============================

speedups = [tempos_strong[0] / t for t in tempos_strong]

plt.figure()
plt.plot(THREADS, speedups, marker='o')
plt.plot(THREADS, THREADS, linestyle='--')  # linha ideal
plt.xlabel("Número de Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Número de Threads")
plt.grid()

plt.savefig("grafico_speedup.png")
plt.close()


# ==============================
# 🔹 GRÁFICO DE EFICIÊNCIA
# ==============================

eficiencias = [s / t for s, t in zip(speedups, THREADS)]

plt.figure()
plt.plot(THREADS, eficiencias, marker='o')
plt.xlabel("Número de Threads")
plt.ylabel("Eficiência")
plt.title("Eficiência vs Número de Threads")
plt.grid()

plt.savefig("grafico_eficiencia.png")
plt.close()


# ==============================
# 🔹 GRÁFICO DE WEAK SCALING
# ==============================

plt.figure()
plt.plot(THREADS, tempos_weak, marker='o')
plt.xlabel("Número de Threads")
plt.ylabel("Tempo (s)")
plt.title("Weak Scaling (Tempo vs Threads)")
plt.grid()

plt.savefig("grafico_weak_scaling.png")
plt.close()


print("Gráficos gerados:")
print("- grafico_speedup.png")
print("- grafico_eficiencia.png")
print("- grafico_weak_scaling.png")