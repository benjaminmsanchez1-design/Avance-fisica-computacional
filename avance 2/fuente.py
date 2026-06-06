

import numpy as np


def ricker(f_central, dt, n_steps, t0=None):
    if t0 is None:
        t0 = 1.0 / f_central  # evita truncar el pulso en t=0

    t = np.arange(n_steps) * dt
    arg = (np.pi * f_central * (t - t0)) ** 2
    s = (1.0 - 2.0 * arg) * np.exp(-arg)
    return t, s


if __name__ == "__main__":
    # Prueba rapida: graficar el pulso y su espectro
    import matplotlib.pyplot as plt

    f0 = 25.0
    dt = 0.001
    n = 800
    t, s = ricker(f0, dt, n)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, s, "b")
    ax[0].set_title("Pulso de Ricker")
    ax[0].set_xlabel("Tiempo [s]")
    ax[0].set_ylabel("Amplitud")
    ax[0].grid(True)

    # Espectro de amplitud
    freqs = np.fft.rfftfreq(n, dt)
    espectro = np.abs(np.fft.rfft(s))
    ax[1].plot(freqs, espectro, "r")
    ax[1].set_xlim(0, 100)
    ax[1].set_title("Espectro de amplitud")
    ax[1].set_xlabel("Frecuencia [Hz]")
    ax[1].set_ylabel("|S(f)|")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("fuente_ricker.png", dpi=150)
    plt.show()
