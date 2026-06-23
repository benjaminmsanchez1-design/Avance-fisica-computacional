import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fuente import ricker


#  Parametros

nx, ny = 300, 300
dx = dy = 10.0
dt = 0.001
n_steps = 1000

f_central = 25.0
sx, sy = nx // 2, 30           # fuente

sy_rec = 5
rec_x = np.arange(20, nx - 20, 5)

snap_cada = 10

# PML
N_PML = 30                     # ancho de la PML en nodos
R_coef = 1e-6                  # reflexion teorica deseada



#  CFL 

def verificar_cfl(c_max, dt, dx):
    cfl = c_max * dt / dx
    cfl_max = 0.606
    print(f"Numero CFL = {cfl:.3f}  (limite = {cfl_max})")
    if cfl > cfl_max:
        raise ValueError(f"INESTABLE: CFL={cfl:.3f} > {cfl_max}.")
    print("Esquema ESTABLE.\n")
    return cfl



#  Perfiles de amortiguamiento PML
#  d(x) = d0 * (x/L)^2  ,  d0 = -3 c log(R) / (2 L)

def construir_perfiles_pml(C, N, R):
    """Devuelve d_x(y,x) y d_y(y,x): coeficientes de absorcion."""
    L = N * dx
    c_ref = C.max()
    d0 = -3.0 * c_ref * np.log(R) / (2.0 * L)

    dx_prof = np.zeros(nx)
    dy_prof = np.zeros(ny)

    for i in range(N):
        # distancia normalizada desde el borde interior de la PML
        pos = (N - i) / N
        val = d0 * pos ** 2
        dx_prof[i] = val
        dx_prof[nx - 1 - i] = val
        dy_prof[i] = val
        dy_prof[ny - 1 - i] = val

    # expandir a 2D
    Dx = np.tile(dx_prof, (ny, 1))        # varia en x
    Dy = np.tile(dy_prof[:, None], (1, nx))  # varia en y
    return Dx, Dy



#  Laplaciano parcial de 4to orden (segundas derivadas separadas)

def d2_dx2(u):
    out = np.zeros_like(u)
    out[:, 2:-2] = (
        -u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2]
        + 16 * u[:, 1:-3] - u[:, 0:-4]
    ) / (12 * dx ** 2)
    return out

def d2_dy2(u):
    out = np.zeros_like(u)
    out[2:-2, :] = (
        -u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :]
        + 16 * u[1:-3, :] - u[0:-4, :]
    ) / (12 * dy ** 2)
    return out



#  Solver con PMl split-field

def simular_pml():
    try:
        C = np.load("modelo_velocidad.npy")
        print("Modelo de velocidad cargado.")
    except FileNotFoundError:
        raise FileNotFoundError("Ejecuta primero modelo.py")

    verificar_cfl(C.max(), dt, dx)
    t, src = ricker(f_central, dt, n_steps)

    c2 = C ** 2
    Dx, Dy = construir_perfiles_pml(C, N_PML, R_coef)

    # Campos split
    ux_prev = np.zeros((ny, nx)); ux_curr = np.zeros((ny, nx))
    uz_prev = np.zeros((ny, nx)); uz_curr = np.zeros((ny, nx))

    snapshots = []
    sismograma = np.zeros((n_steps, len(rec_x)))

    # coeficientes para el esquema amortiguado
    #  (1 + d*dt/2) u^{n+1} = 2u^n - (1 - d*dt/2)u^{n-1} + dt^2 c^2 Lap
    ax = 1.0 + Dx * dt / 2.0
    bx = 1.0 - Dx * dt / 2.0
    ay = 1.0 + Dy * dt / 2.0
    by = 1.0 - Dy * dt / 2.0

    print("Iniciando simulacion PML...")
    for n in range(n_steps):
        u = ux_curr + uz_curr      # campo total

        lap_x = d2_dx2(u)
        lap_z = d2_dy2(u)

        ux_next = (2 * ux_curr - bx * ux_prev + dt ** 2 * c2 * lap_x) / ax
        uz_next = (2 * uz_curr - by * uz_prev + dt ** 2 * c2 * lap_z) / ay

        # inyeccion de fuente (repartida 50/50 en ambas componentes)
        ux_next[sy, sx] += 0.5 * dt ** 2 * c2[sy, sx] * src[n]
        uz_next[sy, sx] += 0.5 * dt ** 2 * c2[sy, sx] * src[n]

        u_total = ux_next + uz_next
        sismograma[n, :] = u_total[sy_rec, rec_x]

        if n % snap_cada == 0:
            snapshots.append(u_total.copy())

        ux_prev, ux_curr = ux_curr, ux_next
        uz_prev, uz_curr = uz_curr, uz_next

        if n % 100 == 0:
            print(f"  paso {n}/{n_steps}")

    print("Simulacion PML terminada.\n")
    return np.array(snapshots), sismograma, t, C



#  Visualizacion
def graficar_snapshots(snapshots, C, tiempos_idx=(20, 50, 80)):
    fig, axes = plt.subplots(1, len(tiempos_idx), figsize=(15, 5))
    extent = [0, nx * dx, ny * dy, 0]
    vmax = np.max(np.abs(snapshots)) * 0.1
    for ax, idx in zip(axes, tiempos_idx):
        ax.imshow(C, cmap="gray", extent=extent, aspect="auto", alpha=0.3)
        ax.imshow(snapshots[idx], cmap="seismic", extent=extent,
                  aspect="auto", vmin=-vmax, vmax=vmax, alpha=0.7)
        ax.set_title(f"t = {idx * snap_cada * dt * 1000:.0f} ms")
        ax.set_xlabel("x [m]"); ax.set_ylabel("Profundidad [m]")
    plt.tight_layout()
    plt.savefig("snapshots_pml.png", dpi=150)
    plt.show()

def graficar_sismograma(sismograma, t):
    plt.figure(figsize=(8, 7))
    extent = [0, len(rec_x), t[-1], 0]
    vmax = np.max(np.abs(sismograma)) * 0.3
    plt.imshow(sismograma, cmap="gray", aspect="auto",
               extent=extent, vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Amplitud")
    plt.title("Sismograma con PML")
    plt.xlabel("Numero de receptor"); plt.ylabel("Tiempo [s]")
    plt.savefig("sismograma_pml.png", dpi=150)
    plt.show()

def crear_animacion(snapshots, C, archivo="propagacion_pml.gif"):
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [0, nx * dx, ny * dy, 0]
    vmax = np.max(np.abs(snapshots)) * 0.1
    ax.imshow(C, cmap="gray", extent=extent, aspect="auto", alpha=0.3)
    im = ax.imshow(snapshots[0], cmap="seismic", extent=extent,
                   aspect="auto", vmin=-vmax, vmax=vmax, alpha=0.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("Profundidad [m]")
    titulo = ax.set_title("t = 0 ms")
    def update(frame):
        im.set_array(snapshots[frame])
        titulo.set_text(f"t = {frame * snap_cada * dt * 1000:.0f} ms")
        return [im, titulo]
    anim = animation.FuncAnimation(fig, update, frames=len(snapshots),
                                   interval=50, blit=False)
    anim.save(archivo, writer="pillow", fps=20)
    print(f"Animacion guardada en '{archivo}'")
    plt.close()


if __name__ == "__main__":
    snapshots, sismograma, t, C = simular_pml()
    graficar_snapshots(snapshots, C)
    graficar_sismograma(sismograma, t)
    crear_animacion(snapshots, C)
    # Guardamos el sismograma para la tomografia
    np.save("sismograma_pml.npy", sismograma)
    np.save("tiempo.npy", t)
    print("Resultados PML generados.")
