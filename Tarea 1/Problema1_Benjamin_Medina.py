import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress

# ==============================================================================
# INCISO (B): Construcción del Hamiltoniano
# ==============================================================================
# 1. Definimos las matrices de Pauli básicas para un solo espín (matrices 2x2)
def matrices_pauli():
    # Matriz sigma_x: 
    sx = np.array([[0, 1], 
                   [1, 0]], dtype=complex)
    
    # Matriz sigma_z: 
    sz = np.array([[1, 0], 
                   [0, -1]], dtype=complex)
    
    # Matriz Identidad:
    I = np.eye(2, dtype=complex)
    
    return sx, sz, I

# 2. Función principal para construir el Hamiltoniano
def construir_hamiltoniano(N, J, B):
    sx, sz, I = matrices_pauli()
    
    # La dimensión total del espacio de Hilbert es 2 elevado a N
    dim = 2**N
    
    # Iniciamos el Hamiltoniano como una matriz llena de ceros de tamaño (dim x dim)
    H = np.zeros((dim, dim), dtype=complex)
    
    # --- PRIMERA PARTE: Término de interacción J (sigma_x en i y sigma_x en i+1) ---
    for i in range(N - 1):
        # Creamos una lista donde cada elemento es la Identidad, simulando los N espines
        operadores = [I] * N
        
        # En la posición 'i' y la vecina 'i+1', reemplazamos la Identidad por sigma_x
        operadores[i] = sx
        operadores[i+1] = sx
        
        # Ahora construimos el producto tensorial de toda la cadena
        # Empezamos con el primer operador de la lista
        termino = operadores[0]
        # Multiplicamos tensorialmente (np.kron) por los demás operadores de la cadena
        for j in range(1, N):
            termino = np.kron(termino, operadores[j])
            
        # Sumamos este término multiplicado por J al Hamiltoniano total
        H += J * termino
        
    # --- SEGUNDA PARTE: Término de campo transversal B (sigma_z en i) ---
    for i in range(N):
        operadores = [I] * N
        
        # Reemplazamos la Identidad por sigma_z solo en el espín 'i'
        operadores[i] = sz
        
        termino = operadores[0]
        for j in range(1, N):
            termino = np.kron(termino, operadores[j])
            
        # Sumamos este término multiplicado por B al Hamiltoniano total
        H += B * termino
        
    return H

# ==============================================================================
# INCISO (C): Evolución Temporal y Probabilidad de Retorno
# ==============================================================================
print("\nEjecutando Inciso (c): Evolución temporal...")
# 1. Configuramos los parámetros de la simulación
N_sim = 4       # Número de espines
J = 1.0         # Fijamos J en 1.0 como referencia
# Definimos los casos pedidos: B/J << 1, B/J = 1, y B/J >> 1
B_valores = [0.1, 1.0, 10.0] 

# Creamos el arreglo de tiempo: 100 pasos desde t=0 hasta t=10
t_arr = np.linspace(0, 10, 100)
dt = t_arr[1] - t_arr[0] # Diferencia de tiempo entre cada paso (Delta t)

# 2. Construcción del estado inicial |Psi(0)>
# El enunciado indica que el estado es "spin down" para todas las partículas.
# Un spin down se representa como el vector columna [0, 1] (transpuesto).
psi_down = np.array([0, 1], dtype=complex)
psi_0 = psi_down

# Multiplicamos tensorialmente para tener N_sim partículas en estado "down"
for _ in range(1, N_sim):
    psi_0 = np.kron(psi_0, psi_down)

# 3. Configuración del gráfico
plt.figure(figsize=(10, 6))

# Iteramos sobre cada uno de los valores de campo B pedidos
for B in B_valores:
    # Construimos el Hamiltoniano para este valor de B
    H = construir_hamiltoniano(N_sim, J, B)
    
    # Calculamos el operador de evolución temporal U = exp(-i * H * dt)
    # Usamos scipy.linalg.expm que es especial para exponenciar matrices
    U = la.expm(-1j * H * dt)
    
    # Hacemos una copia del estado inicial para evolucionarlo sin perder el original
    psi_t = psi_0.copy()
    probabilidades = [] # Lista para guardar las probabilidades en cada paso
    
    # 4. Evolución temporal paso a paso
    for t in t_arr:
        # np.vdot hace el producto interno <Psi(0)|Psi(t)>. 
        # La probabilidad es el valor absoluto al cuadrado de ese producto interno.
        amplitud = np.vdot(psi_0, psi_t)
        prob = np.abs(amplitud)**2
        probabilidades.append(prob)
        
        # Evolucionamos el estado un paso temporal multiplicándolo por U (U @ psi_t)
        psi_t = U @ psi_t
        
    # Graficamos la curva para este valor específico de B
    plt.plot(t_arr, probabilidades, label=f'B = {B} (B/J = {B/J})')

# Detalles finales del gráfico
plt.title("Probabilidad de retornar al estado inicial $p(t) = |\\langle\\Psi(0)|\\Psi(t)\\rangle|^2$")
plt.xlabel("Tiempo (t)")
plt.ylabel("Probabilidad p(t)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ==============================================================================
# INCISOS (D) y (E): Tiempos de ejecución y Gráficos
# ==============================================================================
print("\nEjecutando Incisos (d) y (e): Midiendo tiempos de cálculo...")
# 1. Definimos los tamaños del sistema que vamos a evaluar
N_lista = [4, 5, 6, 7, 8]
tiempos_ejecucion = [] # Aquí guardaremos cuánto demoró cada N

J_fijo = 1.0
B_fijo = 1.0

# 2. Bucle para medir el tiempo de cada N
for n in N_lista:
    # time.time() guarda la hora exacta (en segundos) en la que empieza este bloque
    inicio = time.time()
    
    # Acción 1: Construir el Hamiltoniano
    H = construir_hamiltoniano(n, J_fijo, B_fijo)
    
    # Acción 2: Diagonalizar el Hamiltoniano
    # la.eigh es una función optimizada de scipy para matrices Hermitianas (como H)
    # Devuelve los valores propios (evals) y los vectores propios (evecs)
    evals, evecs = la.eigh(H) 
    
    # Calculamos el tiempo transcurrido restando la hora final menos la hora inicial
    tiempo_total = time.time() - inicio
    tiempos_ejecucion.append(tiempo_total)
    
    print(f"Para N={n}, la dimensión es {2**n}x{2**n}. Tiempo: {tiempo_total:.4f} segundos")

# 3. Gráfico de Tiempo vs N
plt.figure(figsize=(8, 5))

# Usamos marcadores 'o-' para ver un círculo en cada punto de dato medido
plt.plot(N_lista, tiempos_ejecucion, marker='o', linestyle='-', color='red')

plt.title("Tiempo de Ejecución vs Número de Espines (N)")
plt.xlabel("Número de Espines (N)")
plt.ylabel("Tiempo de Ejecución (segundos)")

# Usamos escala logarítmica en el eje Y para poder ver bien la curva, 
# ya que el crecimiento del tiempo es exponencial y los valores grandes aplastarían a los pequeños.
plt.yscale('log') 

plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.show()