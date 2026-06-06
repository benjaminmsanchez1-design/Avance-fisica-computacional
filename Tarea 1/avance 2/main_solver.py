

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fuente import ricker



# 1. PARAMETROS DE LA SIMULACION

nx, ny = 300, 300        # puntos de grilla (x horizontal, y profundidad)
dx = dy = 10.0           # espaciamiento [m]
dt = 0.001               # paso temporal [s]
n_steps = 1000           # numero de pasos de tiempo

# Fuente
f_central = 25.0         # Hz
sx, sy = nx // 2, 30     # posicion fuente (cerca de superficie)

# Receptores en superficie (fila sy_rec)
sy_rec = 5
rec_x = np.arange(20, nx - 20, 5)   # receptores espaciados

# Guardado de snapshots para animacion
snap_cada = 10           # guardar 1 de cada N frames


# 2. CONDICION DE ESTABILIDAD CFL

def verificar_cfl(c_max, dt, dx):
    
    cfl = c_max * dt / dx
    cfl_max = 0.606  # limite teorico para 4to orden en 2D
    print(f"Numero CFL = {cfl:.3f}  (limite = {cfl_max})")
    if cfl > cfl_max:
        raise ValueError(
            f"INESTABLE: CFL={cfl:.3f} > {cfl_max}. "
            f"Reduce dt o aumenta dx."
        )
    print("Esquema ESTABLE.\n")
    return cfl



# 3. BORDES ABSORBENTES (SPONGE LAYER)

def crear_sponge(nx, ny, ancho=30, factor=0.92):
    sponge = np.ones((ny, nx))
    
    # invertimos para que el borde absorba mas que el interior
    perfil = factor ** ((ancho - np.arange(ancho)) ** 2 / ancho)

    for i in range(ancho):
        amort = factor ** ((ancho - i) / ancho * 3)
        sponge[i, :] *= amort           # borde superior
        sponge[ny - 1 - i, :] *= amort  # borde inferior
        sponge[:, i] *= amort           # borde izquierdo
        sponge[:, nx - 1 - i] *= amort  # borde derecho
    return sponge



# 4. SOLVER PRINCIPAL

def simular():
    # --- Cargar modelo de velocidad desde archivo ---
    try:
        C = np.load("modelo_velocidad.npy")
        print("Modelo de velocidad cargado desde 'modelo_velocidad.npy'")
    except FileNotFoundError:
        raise FileNotFoundError(
            "No existe 'modelo_velocidad.npy'. Ejecuta primero modelo.py"
        )

    # Verificar estabilidad con la velocidad maxima
    verificar_cfl(C.max(), dt, dx)

    # Fuente de Ricker 
    t, src = ricker(f_central, dt, n_steps)

    # Campos de onda: condiciones iniciales = 0 (reposo) 
    u_prev = np.zeros((ny, nx))   # u en t-dt
    u_curr = np.zeros((ny, nx))   # u en t
    u_next = np.zeros((ny, nx))   # u en t+dt

    # Sponge layer 
    sponge = crear_sponge(nx, ny, ancho=30, factor=0.95)

    # Coeficiente (c*dt/dx)^2 para cada punto (medio heterogeneo) 
    coef = (C * dt / dx) ** 2

    # Estructuras de salida 
    snapshots = []
    sismograma = np.zeros((n_steps, len(rec_x)))

    # Coeficientes del laplaciano de 4to orden:
    #   (-1/12, 4/3, -5/2, 4/3, -1/12)
    print("Iniciando simulacion...")
    for n in range(n_steps):
        # Laplaciano de 4to orden (interior, sin tocar 2 puntos de borde)
        lap = (
            -1.0 / 12.0 * u_curr[2:-2, 4:]      # x+2
            + 4.0 / 3.0 * u_curr[2:-2, 3:-1]    # x+1
            - 5.0 / 2.0 * u_curr[2:-2, 2:-2]    # centro (x)
            + 4.0 / 3.0 * u_curr[2:-2, 1:-3]    # x-1
            - 1.0 / 12.0 * u_curr[2:-2, 0:-4]   # x-2
            - 1.0 / 12.0 * u_curr[4:, 2:-2]     # y+2
            + 4.0 / 3.0 * u_curr[3:-1, 2:-2]    # y+1
            - 5.0 / 2.0 * u_curr[2:-2, 2:-2]    # centro (y)
            + 4.0 / 3.0 * u_curr[1:-3, 2:-2]    # y-1
            - 1.0 / 12.0 * u_curr[0:-4, 2:-2]   # y-2
        )

        # Actualizacion temporal (2do orden)
        u_next[2:-2, 2:-2] = (
            2.0 * u_curr[2:-2, 2:-2]
            - u_prev[2:-2, 2:-2]
            + coef[2:-2, 2:-2] * lap
        )

        # Inyeccion de la fuente (termino fuente)
        u_next[sy, sx] += coef[sy, sx] * src[n]

        # Aplicar bordes absorbentes
        u_next *= sponge
        u_curr *= sponge

        # Registrar sismograma (receptores en superficie)
        sismograma[n, :] = u_next[sy_rec, rec_x]

        # Guardar snapshot
        if n % snap_cada == 0:
            snapshots.append(u_next.copy())

        # Rotar campos para el siguiente paso
        u_prev, u_curr, u_next = u_curr, u_next, u_prev

        if n % 100 == 0:
            print(f"  paso {n}/{n_steps}")

    print("Simulacion terminada.\n")
    return np.array(snapshots), sismograma, t, C



# 5. VISUALIZACION

def graficar_snapshots(snapshots, C, tiempos_idx=(20, 50, 80)):
    """Genera figura con snapshots en distintos instantes."""
    fig, axes = plt.subplots(1, len(tiempos_idx), figsize=(15, 5))
    extent = [0, nx * dx, ny * dy, 0]
    vmax = np.max(np.abs(snapshots)) * 0.1

    for ax, idx in zip(axes, tiempos_idx):
        ax.imshow(C, cmap="gray", extent=extent, aspect="auto", alpha=0.3)
        ax.imshow(
            snapshots[idx], cmap="seismic", extent=extent,
            aspect="auto", vmin=-vmax, vmax=vmax, alpha=0.7
        )
        ax.set_title(f"t = {idx * snap_cada * dt * 1000:.0f} ms")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Profundidad [m]")

    plt.tight_layout()
    plt.savefig("snapshots.png", dpi=150)
    plt.show()


def graficar_sismograma(sismograma, t):
    plt.figure(figsize=(8, 7))
    extent = [0, len(rec_x), t[-1], 0]
    vmax = np.max(np.abs(sismograma)) * 0.3
    plt.imshow(
        sismograma, cmap="gray", aspect="auto",
        extent=extent, vmin=-vmax, vmax=vmax
    )
    plt.colorbar(label="Amplitud")
    plt.title("Sismograma (receptores en superficie)")
    plt.xlabel("Numero de receptor")
    plt.ylabel("Tiempo [s]")
    plt.savefig("sismograma.png", dpi=150)
    plt.show()


def crear_animacion(snapshots, C, archivo="propagacion.gif"):
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [0, nx * dx, ny * dy, 0]
    vmax = np.max(np.abs(snapshots)) * 0.1

    ax.imshow(C, cmap="gray", extent=extent, aspect="auto", alpha=0.3)
    im = ax.imshow(
        snapshots[0], cmap="seismic", extent=extent,
        aspect="auto", vmin=-vmax, vmax=vmax, alpha=0.7
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Profundidad [m]")
    titulo = ax.set_title("t = 0 ms")

    def update(frame):
        im.set_array(snapshots[frame])
        titulo.set_text(f"t = {frame * snap_cada * dt * 1000:.0f} ms")
        return [im, titulo]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=50, blit=False
    )
    anim.save(archivo, writer="pillow", fps=20)
    print(f"Animacion guardada en '{archivo}'")
    plt.close()



# 6. PROGRAMA PRINCIPAL

if __name__ == "__main__":
    snapshots, sismograma, t, C = simular()

    graficar_snapshots(snapshots, C)
    graficar_sismograma(sismograma, t)
    crear_animacion(snapshots, C)

    print("Todos los resultados generados con exito.")
