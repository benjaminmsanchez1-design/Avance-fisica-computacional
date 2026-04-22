import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import maxwell
import warnings

warnings.filterwarnings("ignore") 

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y CONSTANTES
# ==============================================================================
N = 125                  
m = 3.32e-27             
kB = 1.380649e-23        
dt = 1e-12               

# Parámetros de Lennard-Jones para H2
epsilon = 33.3 * kB      # Convertimos de K a Joules
sigma = 0.296e-9         # 0.296 nm a metros

T_inicial = 300.0        
L_inicial = 10e-6        

parametros = {'T': T_inicial, 'L': L_inicial}

# ==============================================================================
# 2. INICIALIZACIÓN DEL SISTEMA
# ==============================================================================
# Tarea 1: Red cúbica simple para evitar superposiciones 
n_lado = int(np.round(N**(1/3))) # Raíz cúbica de 125 es 5
espaciado = parametros['L'] / n_lado

posiciones = []
for i in range(n_lado):
    for j in range(n_lado):
        for k in range(n_lado):
            # Posición base en el centro de cada celda + pequeña perturbación
            x = (i + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
            y = (j + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
            z = (k + 0.5) * espaciado + np.random.uniform(-0.1, 0.1) * espaciado
            posiciones.append([x, y, z])
            
posiciones = np.array(posiciones)

# Tarea 2: Inicializar velocidades a temperatura T0
desviacion_v = np.sqrt(kB * parametros['T'] / m)
velocidades = np.random.normal(0, desviacion_v, (N, 3))
# ==============================================================================
# 3. CONFIGURACIÓN DE LOS GRÁFICOS (INTERFAZ EN TIEMPO REAL)
# ==============================================================================
plt.ion() 
fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title('Dinámica Molecular - Gas Real (Lennard-Jones)')

ax_temp = fig.add_subplot(221)
ax_pres = fig.add_subplot(222)
ax_ener = fig.add_subplot(223)
ax_hist = fig.add_subplot(224)

plt.subplots_adjust(bottom=0.25, hspace=0.4)

ax_slider_T = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_T = Slider(ax_slider_T, 'Temperatura (K)', 100.0, 1000.0, valinit=T_inicial)

ax_slider_L = plt.axes([0.15, 0.05, 0.65, 0.03])
slider_L = Slider(ax_slider_L, 'Caja (um)', 0.1, 20.0, valinit=L_inicial*1e6)

def actualizar_T(val):
    T_nueva = slider_T.val
    factor_escala = np.sqrt(T_nueva / parametros['T'])
    global velocidades
    velocidades *= factor_escala
    parametros['T'] = T_nueva

def actualizar_L(val):
    parametros['L'] = slider_L.val * 1e-6 
    global posiciones
    posiciones = np.clip(posiciones, 0, parametros['L'])

slider_T.on_changed(actualizar_T)
slider_L.on_changed(actualizar_L)

# ==============================================================================
# 4. BUCLE DE SIMULACIÓN (DINÁMICA MOLECULAR CON FUERZAS)
# ==============================================================================
historia_T = []
historia_P = []
historia_Ek = []
historia_Ep = []
historia_Etot = []

print("Simulación de Gas Real en ejecución... Cierra la ventana gráfica para detenerla.")

paso = 0
while plt.fignum_exists(fig.number):
    
    L_actual = parametros['L']
    
    # --- CÁLCULO DE FUERZAS DE LENNARD-JONES (VECTORIZADO) ---
    # Calculamos la matriz de diferencias de posición de todas contra todas
    # delta tiene forma (N, N, 3)
    delta = posiciones[:, np.newaxis, :] - posiciones[np.newaxis, :, :]
    
    # Distancia al cuadrado r^2
    r2 = np.sum(delta**2, axis=-1)
    
    # Llenamos la diagonal con infinito para que una partícula no interactúe consigo misma
    np.fill_diagonal(r2, np.inf)
    
    # SEGURO NUMÉRICO: Limitamos la distancia mínima para que la fuerza no explote 
    # por culpa del gran paso temporal dt = 1 ps.
    r2 = np.maximum(r2, (0.8 * sigma)**2)
    
    # Cálculos optimizados para L-J
    sr2 = (sigma**2) / r2
    sr6 = sr2**3
    sr12 = sr6**2
    
    # Magnitud de la fuerza dividida por r (para multiplicarla por el vector delta luego)
    f_mag = 24 * epsilon * (2 * sr12 - sr6) / r2
    
    # Sumamos las fuerzas sobre cada partícula a lo largo del eje 1
    fuerzas = np.sum(f_mag[:, :, np.newaxis] * delta, axis=1)
    
    # --- INTEGRACIÓN DE LAS ECUACIONES (MÉTODO DE EULER CROMER) ---
    # Primero actualizamos velocidad con la fuerza, luego posición
    velocidades += (fuerzas / m) * dt
    posiciones += velocidades * dt
    
    # --- COLISIONES ELÁSTICAS CON LAS PAREDES ---
    chocan_izq = posiciones < 0
    chocan_der = posiciones > L_actual
    velocidades[chocan_izq] *= -1
    velocidades[chocan_der] *= -1
    posiciones = np.clip(posiciones, 0, L_actual)
    
    # --- CÁLCULO DE PROPIEDADES TERMODINÁMICAS ---
    energia_cinetica = 0.5 * m * np.sum(velocidades**2)
    
    # Energía potencial total (dividimos por 2 para no contar pares dobles)
    energia_potencial = 4 * epsilon * np.sum(sr12 - sr6) / 2.0
    
    energia_total = energia_cinetica + energia_potencial
    
    T_actual = (2.0 * energia_cinetica) / (3.0 * N * kB)
    volumen = L_actual**3
    P_actual = (N * kB * T_actual) / volumen
    
    historia_T.append(T_actual)
    historia_P.append(P_actual)
    historia_Ek.append(energia_cinetica)
    historia_Ep.append(energia_potencial)
    historia_Etot.append(energia_total)
    
    if len(historia_T) > 1000:
        historia_T.pop(0)
        historia_P.pop(0)
        historia_Ek.pop(0)
        historia_Ep.pop(0)
        historia_Etot.pop(0)
        
    # --- ACTUALIZACIÓN DE GRÁFICOS ---
    if paso % 15 == 0:
        ax_temp.clear()
        ax_temp.plot(historia_T, color='red')
        ax_temp.set_title("Temperatura $T(t)$")
        ax_temp.set_ylabel("Kelvin (K)")
        
        ax_pres.clear()
        ax_pres.plot(historia_P, color='blue')
        ax_pres.set_title("Presión $P(t)$")
        ax_pres.set_ylabel("Pascales (Pa)")
        
        ax_ener.clear()
        ax_ener.plot(historia_Ek, color='green', label='Cinética')
        ax_ener.plot(historia_Ep, color='purple', label='Potencial')
        ax_ener.plot(historia_Etot, color='black', label='Total', linestyle='--')
        ax_ener.set_title("Energías")
        ax_ener.set_ylabel("Joules (J)")
        ax_ener.legend(loc="upper right", fontsize=8)
        
        ax_hist.clear()
        rapidez = np.linalg.norm(velocidades, axis=1)
        ax_hist.hist(rapidez, bins=12, density=True, color='gray', alpha=0.7, label='Simulación')
        
        v_teo = np.linspace(0, np.max(rapidez)*1.5, 100)
        param_a = np.sqrt(kB * T_actual / m)
        ax_hist.plot(v_teo, maxwell.pdf(v_teo, scale=param_a), 'k-', lw=2, label='Teoría M-B')
        ax_hist.set_title("Distribución de Velocidades")
        ax_hist.legend()
        ax_hist.set_xlim(0, 15000)
        plt.draw()
        plt.pause(0.001)
        
    paso += 1

print("Simulación terminada.")