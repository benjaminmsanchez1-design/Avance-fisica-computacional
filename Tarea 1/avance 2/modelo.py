
import numpy as np


def construir_modelo(nx, ny, dx, dy, capas, archivo="modelo_velocidad.npy"):
    C = np.zeros((ny, nx))

    # Para cada fila (profundidad) asignamos la velocidad de su capa
    for j in range(ny):
        profundidad = j * dy
        vel = capas[0][1]  # valor por defecto = primera capa
        for tope, v in capas:
            if profundidad >= tope:
                vel = v
        C[j, :] = vel

    np.save(archivo, C)
    print(f"Modelo guardado en '{archivo}'  -->  shape={C.shape}")
    print(f"Velocidades: min={C.min():.0f}  max={C.max():.0f} m/s")
    return C


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx, ny = 300, 300
    dx, dy = 10.0, 10.0

    # (profundidad_tope_en_m, velocidad_m/s)
    capas = [
        (0.0,    2000.0),   # capa superficial
        (1000.0, 3000.0),   # capa intermedia
        (2000.0, 4000.0),   # capa profunda
    ]

    C = construir_modelo(nx, ny, dx, dy, capas)

    # Visualizar el modelo
    plt.figure(figsize=(7, 6))
    extent = [0, nx * dx, ny * dy, 0]  # y invertido: profundidad hacia abajo
    plt.imshow(C, cmap="viridis", extent=extent, aspect="auto")
    plt.colorbar(label="Velocidad [m/s]")
    plt.title("Modelo de velocidad por capas")
    plt.xlabel("Distancia x [m]")
    plt.ylabel("Profundidad z [m]")
    plt.savefig("modelo_velocidad.png", dpi=150)
    plt.show()
