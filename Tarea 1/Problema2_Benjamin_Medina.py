import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress

# ==============================================================================
# INCISO (A): Generación de la señal
# ==============================================================================
def generar_senal(N):
    """
    Genera la señal discreta x_n = sin(2*pi*f1*t_n) + 0.5*sin(2*pi*f2*t_n)
    """
    # Elegimos dos frecuencias arbitrarias para la señal (ej. 5 Hz y 15 Hz)
    f1 = 5.0
    f2 = 15.0
    
    # np.linspace crea un arreglo de 'N' tiempos igualmente espaciados entre 0 y 1 segundo.
    # endpoint=False asegura que no incluyamos el último punto para que sea un ciclo exacto.
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Evaluamos la fórmula matemática dada en el enunciado
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    return t, x

# ==============================================================================
# INCISO (B): Implementación de la DFT Directa
# ==============================================================================
def dft_directa(x):
    """
    Calcula la Transformada Discreta de Fourier usando la definición matemática (dos ciclos for)
    """
    N = len(x) # Obtenemos el tamaño de la señal
    
    # Creamos un arreglo vacío de números complejos (dtype=complex) para guardar el resultado X_k
    X = np.zeros(N, dtype=complex)
    
    # Primer ciclo for: itera sobre cada frecuencia 'k' que queremos calcular
    for k in range(N):
        # Segundo ciclo for: itera sobre cada punto 'n' de la señal original en el tiempo
        for n in range(N):
            # Aplicamos la fórmula sumando la contribución de cada punto
            # np.exp(-2j * ...) es la forma de escribir e^(-i * ...) en Python. 'j' es el imaginario.
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
            
    return X

# ==============================================================================
# INCISOS (C) Y (D): Gráfico del Espectro y uso de FFT
# ==============================================================================
print("Calculando espectros para Incisos C y D...")
N_test = 1000 # Usamos 1000 puntos para este gráfico de prueba
t_test, x_test = generar_senal(N_test)

# Calculamos la transformada con nuestra función (Inciso C)
X_dft = dft_directa(x_test)

# Calculamos la transformada con el algoritmo rápido de Numpy (Inciso D)
X_fft = np.fft.fft(x_test)

# Graficamos el espectro |X_k|
plt.figure(figsize=(10, 5))

# np.abs() saca el módulo del número complejo (la amplitud de la frecuencia)
# Solo graficamos la primera mitad [:N_test//2] porque la FFT de una señal real es simétrica (espejo)
frecuencias = np.arange(N_test // 2)
plt.plot(frecuencias, np.abs(X_dft[:N_test//2]), label="DFT Directa (nuestra)", linestyle='--', linewidth=3)
plt.plot(frecuencias, np.abs(X_fft[:N_test//2]), label="FFT (Numpy)", alpha=0.7)

plt.title("Espectro de Frecuencias $|X_k|$ (Incisos C y D)")
plt.xlabel("Índice de Frecuencia k")
plt.ylabel("Amplitud $|X_k|$")
# Hacemos zoom (limite en x) para ver bien los dos picos (que deberían estar en 5 y 15)
plt.xlim(0, 30) 
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# INCISOS (E), (F) Y (G): Comparación de Tiempos y Complejidad
# ==============================================================================
print("\nIniciando medición de tiempos (Incisos E, F, G)... esto puede tomar unos segundos.")
N_valores = [100, 1000, 10000, 100000]

tiempos_dft = []
tiempos_fft = []
N_dft = [] 

for n in N_valores:
    _, x_n = generar_senal(n)
    
    # 1. Medimos tiempo de la FFT (Numpy)
    inicio = time.time()
    np.fft.fft(x_n)
    tiempos_fft.append(time.time() - inicio)
    
    # 2. Medimos tiempo de la DFT Directa
    # SEGURO: Evitamos calcular N=100000 en la DFT para que no se congele el PC-
    if n <= 10000:
        inicio = time.time()
        dft_directa(x_n)
        tiempos_dft.append(time.time() - inicio)
        N_dft.append(n)
        print(f"N={n} -> DFT: {tiempos_dft[-1]:.4f}s | FFT: {tiempos_fft[-1]:.6f}s")
    else:
        print(f"N={n} -> DFT: Omitido (tomaría demasiado) | FFT: {tiempos_fft[-1]:.6f}s")

# --- Gráficos de Tiempo ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico lineal (Inciso F)
ax1.plot(N_dft, tiempos_dft, 'o-', label="DFT Directa")
ax1.plot(N_valores, tiempos_fft, 's-', label="FFT")
ax1.set_title("Tiempo de Ejecución vs N (Escala Lineal)")
ax1.set_xlabel("Tamaño de la señal (N)")
ax1.set_ylabel("Tiempo (s)")
ax1.legend()
ax1.grid(True)

# Gráfico Log-Log (Inciso G)
ax2.loglog(N_dft, tiempos_dft, 'o-', label="DFT Directa")
ax2.loglog(N_valores, tiempos_fft, 's-', label="FFT")
ax2.set_title("Tiempo de Ejecución vs N (Escala Log-Log)")
ax2.set_xlabel("Tamaño de la señal (N)")
ax2.set_ylabel("Tiempo (s)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- Cálculo experimental del exponente de escalamiento (Inciso G) ---
# Usamos regresión lineal sobre los logaritmos: log(t) = exponente * log(N) + constante
log_N_dft = np.log10(N_dft)
log_t_dft = np.log10(tiempos_dft)
exponente_dft, _, _, _, _ = linregress(log_N_dft, log_t_dft)

log_N_fft = np.log10(N_valores)
log_t_fft = np.log10(tiempos_fft)
exponente_fft, _, _, _, _ = linregress(log_N_fft, log_t_fft)

print("\n--- Resultados Inciso (g) ---")
print(f"Exponente experimental DFT Directa: {exponente_dft:.2f} (Teórico esperado: ~2.0)")
print(f"Exponente experimental FFT: {exponente_fft:.2f} (Teórico esperado: ~1.0, escala como N log N)")

# ==============================================================================
# INCISO (H): ¿Cuándo la FFT es 100 veces más rápida?
# ==============================================================================
print("\n--- Resultados Inciso (h) ---")
# Comparamos punto a punto usando las listas que tienen el mismo tamaño
for i in range(len(N_dft)):
    razon = tiempos_dft[i] / tiempos_fft[i]
    print(f"Para N={N_dft[i]}, la FFT es {razon:.1f} veces más rápida.")
    if razon >= 100:
        print(f"-> ¡El umbral de 100x se supera cerca de N = {N_dft[i]}!")