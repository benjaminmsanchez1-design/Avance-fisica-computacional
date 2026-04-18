import numpy as np
import matplotlib.pyplot as plt
from fuente import pulso_ricker

nx, ny = 200, 200
dx, dy = 10.0, 10.0
dt = 0.001
n_steps = 800

C = np.ones((nx, ny)) * 2000.0
alpha = (C * dt)**2

u_prev = np.zeros((nx, ny))
u_curr = np.zeros((nx, ny))
u_next = np.zeros((nx, ny))

frecuencia_central = 20.0
t0 = 1.0 / frecuencia_central
fuente = pulso_ricker(frecuencia_central, dt, n_steps, t0)

sx, sy = nx // 2, ny // 2

for n in range(1, n_steps):
    u_curr[sx, sy] += fuente[n]
    
    laplaciano_x = (- u_curr[4:, 2:-2] + 16 * u_curr[3:-1, 2:-2] - 30 * u_curr[2:-2, 2:-2] + 16 * u_curr[1:-3, 2:-2] - u_curr[:-4, 2:-2]) / (12.0 * dx**2)
    laplaciano_y = (- u_curr[2:-2, 4:] + 16 * u_curr[2:-2, 3:-1] - 30 * u_curr[2:-2, 2:-2] + 16 * u_curr[2:-2, 1:-3] - u_curr[2:-2, :-4]) / (12.0 * dy**2)
    
    laplaciano = laplaciano_x + laplaciano_y

    u_next[2:-2, 2:-2] = 2 * u_curr[2:-2, 2:-2] - u_prev[2:-2, 2:-2] + alpha[2:-2, 2:-2] * laplaciano

    u_prev[:, :] = u_curr[:, :]
    u_curr[:, :] = u_next[:, :]

plt.imshow(u_curr, cmap='seismic', vmin=-np.max(u_curr)*0.1, vmax=np.max(u_curr)*0.1)
plt.colorbar(label="Amplitud de la onda")
plt.title(f"Frente de onda sísmica a T={n_steps*dt:.3f} s")
plt.xlabel("Distancia X")
plt.ylabel("Profundidad Z")
plt.show()
