
import numpy as np
import matplotlib.pyplot as plt


#  Parametros

nx, ny = 300, 300
dx = dy = 10.0
dt = 0.001

sx_nodo, sy_nodo = nx // 2, 30        # fuente
sy_rec = 5
rec_x = np.arange(20, nx - 20, 5)

# Geometria fisica
fuente_xz = np.array([sx_nodo * dx, sy_nodo * dy])
rec_xz = np.column_stack([rec_x * dx, np.full(len(rec_x), sy_rec * dy)])

# Discretizacion del modelo a invertir: 3 capas horizontales
#  topes de las capas en profundidad [m]
topes_capas = [0.0, 1000.0, 2000.0]
prof_max = ny * dy



#  1. Extraccion de tiempos de llegada

def extraer_tiempos_llegada(sismograma, t, umbral_frac=0.05):
    n_steps, n_rec = sismograma.shape
    tiempos = np.zeros(n_rec)
    for r in range(n_rec):
        traza = np.abs(sismograma[:, r])
        pico = traza.max()
        if pico <= 0:
            tiempos[r] = np.nan
            continue
        umbral = umbral_frac * pico
        idx = np.argmax(traza > umbral)   # primer cruce del umbral
        tiempos[r] = t[idx]
    return tiempos



#  2 Trazado de rayos rectos.

def construir_matriz_rayos(fuente, receptores, topes):
    n_rayos = len(receptores)
    n_capas = len(topes)
    G = np.zeros((n_rayos, n_capas))

    # limites en profundidad de cada capa
    bordes = topes + [prof_max]   # [0,1000,2000,3000]

    for i, rec in enumerate(receptores):
        x0, z0 = fuente
        x1, z1 = rec
        L_total = np.hypot(x1 - x0, z1 - z0)

        # muestreamos el rayo finamente y acumulamos longitud por capa
        n_muestras = 2000
        s = np.linspace(0, 1, n_muestras)
        zs = z0 + s * (z1 - z0)
        ds = L_total / (n_muestras - 1)

        for k in range(n_capas):
            ztop = bordes[k]
            zbot = bordes[k + 1]
            dentro = (zs >= ztop) & (zs < zbot)
            G[i, k] = np.sum(dentro) * ds

    return G



#  3. Inversion por minimos cuadrados

def invertir_lentitud(G, tiempos, lam=1e-3, m_ref=None):
    # eliminar rayos invalidos
    valido = ~np.isnan(tiempos)
    Gv = G[valido]
    dv = tiempos[valido]

    n_capas = G.shape[1]
    if m_ref is None:
        m_ref = np.zeros(n_capas)

    A = Gv.T @ Gv + lam * np.eye(n_capas)
    b = Gv.T @ dv + lam * m_ref
    m = np.linalg.solve(A, b)
    return m



#  4. Reconstruccion del modelo

def reconstruir_modelo(m_lentitud, topes):
    v_capas = 1.0 / m_lentitud
    C_inv = np.zeros((ny, nx))
    bordes = topes + [prof_max]
    for j in range(ny):
        z = j * dy
        for k in range(len(topes)):
            if bordes[k] <= z < bordes[k + 1]:
                C_inv[j, :] = v_capas[k]
                break
    return C_inv, v_capas



#  programa principal de tomografia

if __name__ == "__main__":
    # cargar datos del solver 
    sismograma = np.load("sismograma_pml.npy")
    t = np.load("tiempo.npy")
    C_real = np.load("modelo_velocidad.npy")

    # 1. tiempos de llegada observados 
    tiempos_obs = extraer_tiempos_llegada(sismograma, t)
    print("Tiempos de llegada (primeros 5):", tiempos_obs[:5])

    #  2. matriz de rayos 
    G = construir_matriz_rayos(fuente_xz, rec_xz, topes_capas)
    print(f"Matriz de rayos G: shape={G.shape}")

    # 3. inversion
    # modelo de referencia: lentitud de una velocidad media (3000 m/s)
    m_ref = np.full(len(topes_capas), 1.0 / 3000.0)
    m_inv = invertir_lentitud(G, tiempos_obs, lam=1e-2, m_ref=m_ref)

    # 4. reconstruir y comparar 
    C_inv, v_inv = reconstruir_modelo(m_inv, topes_capas)

    v_real = [2000.0, 3000.0, 4000.0]
    print("\n=== RESULTADO DE LA INVERSION ===")
    for k, (vr, vi) in enumerate(zip(v_real, v_inv)):
        err = 100 * abs(vi - vr) / vr
        print(f"Capa {k+1}: real={vr:.0f}  invertida={vi:.0f} m/s "
              f"(error {err:.1f}%)")

    #  visualizacion comparativa 
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    extent = [0, nx * dx, ny * dy, 0]

    im0 = axes[0].imshow(C_real, cmap="viridis", extent=extent, aspect="auto")
    axes[0].set_title("Modelo REAL")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("Profundidad [m]")
    # dibujar rayos
    for rec in rec_xz:
        axes[0].plot([fuente_xz[0], rec[0]], [fuente_xz[1], rec[1]],
                     "r-", lw=0.3, alpha=0.4)
    axes[0].plot(fuente_xz[0], fuente_xz[1], "w*", ms=15)
    plt.colorbar(im0, ax=axes[0], label="v [m/s]")

    im1 = axes[1].imshow(C_inv, cmap="viridis", extent=extent, aspect="auto",
                         vmin=C_real.min(), vmax=C_real.max())
    axes[1].set_title("Modelo INVERTIDO (tomografia)")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("Profundidad [m]")
    plt.colorbar(im1, ax=axes[1], label="v [m/s]")

    plt.tight_layout()
    plt.savefig("tomografia_resultado.png", dpi=150)
    plt.show()

    #  perfil 1D de velocidad 
    plt.figure(figsize=(5, 7))
    z_axis = np.arange(ny) * dy
    plt.step(C_real[:, nx//2], z_axis, "k-", lw=2, label="Real")
    plt.step(C_inv[:, nx//2], z_axis, "r--", lw=2, label="Invertido")
    plt.gca().invert_yaxis()
    plt.xlabel("Velocidad [m/s]"); plt.ylabel("Profundidad [m]")
    plt.title("Perfil de velocidad")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("perfil_velocidad.png", dpi=150)
    plt.show()

    print("\nTomografia completada.")
