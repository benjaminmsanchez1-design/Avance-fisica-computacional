import numpy as np

def pulso_ricker(f_central, dt, n_steps, t0):
    t = np.arange(n_steps) * dt
    arg = np.pi * f_central * (t - t0)
    ricker = (1.0 - 2.0 * arg**2) * np.exp(-arg**2)
    return ricker
