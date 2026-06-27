# ============================================================
# PROBLEMA 2: TERAPIA CON PROTONES (Monte Carlo)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(42)

# ------------------------------------------------------------
# Constantes físicas
# ------------------------------------------------------------
mp = 938.272      # masa protón (MeV)
me = 0.511        # masa electrón (MeV)
K  = 0.307075     # MeV cm^2/g
Z_A = 0.5551      # agua
rho = 1.0         # g/cm^3
I  = 75e-6        # potencial excitación agua (MeV)
re = 2.818e-13    # radio clásico electrón (cm)
Ne = 3.343e23     # densidad electrónica del agua (e/cm^3)

# ------------------------------------------------------------
# INCISO (b): Bethe-Bloch
# ------------------------------------------------------------
def bethe_bloch(E_MeV):
    """ -dE/dx (MeV/cm) para protón en agua """
    if E_MeV < 0.1:
        return 0.0
    gamma = 1 + E_MeV/mp
    beta2 = 1 - 1/gamma**2
    if beta2 <= 0:
        return 0.0
    Tmax = (2*me*beta2*gamma**2)/(1 + 2*gamma*me/mp + (me/mp)**2)
    arg = 2*me*beta2*gamma**2*Tmax/I**2
    if arg <= 1:
        return 0.0
    val = K*rho*Z_A/beta2 * (0.5*np.log(arg) - beta2)
    return max(val, 0.0)

# --- Rango CSDA por integración ---
def rango_CSDA(E0):
    """ R_CSDA = integral_0^E0 dE / (-dE/dx) """
    def integrando(E):
        s = bethe_bloch(E)
        return 1.0/s if s > 0 else 0.0
    R, _ = quad(integrando, 0.5, E0, limit=200)
    return R  # en cm

# --- Comparar con NIST PSTAR ---
# Valores tabulados PSTAR (rango CSDA en g/cm^2 = cm para agua rho=1)
nist = {50: 2.227, 150: 15.77, 250: 37.94}  # cm aprox

print("="*50)
print("COMPARACIÓN RANGO CSDA con NIST PSTAR")
print("="*50)
print(f"{'E0 (MeV)':<10}{'R_sim (cm)':<14}{'R_NIST (cm)':<14}{'Error %':<10}")
for E0 in [50, 150, 250]:
    R_sim = rango_CSDA(E0)
    R_nist = nist[E0]
    err = abs(R_sim - R_nist)/R_nist*100
    print(f"{E0:<10}{R_sim:<14.3f}{R_nist:<14.3f}{err:<10.2f}")

# --- Gráfico Bethe-Bloch ---
E_range = np.linspace(0.5, 250, 500)
dEdx = np.array([bethe_bloch(E) for E in E_range])

fig, ax = plt.subplots(figsize=(8,5))
ax.semilogy(E_range, dEdx, 'b-', lw=2)
ax.invert_xaxis()
ax.set_xlabel('Energía cinética (MeV)  → se frena')
ax.set_ylabel('-dE/dx (MeV/cm)')
ax.set_title('Bethe-Bloch: protón en agua')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p2_bethebloch.png', dpi=120, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# INCISO (c): Simular 10^4 protones SIN fluctuación
# ------------------------------------------------------------
def simular_proton(E0, dx_cm, straggling=False):
    """ Simula un protón. Retorna (posiciones, dE_depositada) """
    E = E0
    x = 0.0
    posiciones = []
    energias_dep = []
    while E > 0.5:
        s = bethe_bloch(E)
        if s <= 0:
            break
        dE_medio = s * dx_cm

        if straggling:
            # Bohr: sigma_E^2 = 4 pi re^2 (me c^2)^2 Ne (z^2/beta^2) dx
            gamma = 1 + E/mp
            beta2 = 1 - 1/gamma**2
            sigma2_E = 4*np.pi*re**2*(me)**2*Ne*(1.0/beta2)*dx_cm
            sigma_E = np.sqrt(sigma2_E)
            dE = dE_medio + np.random.normal(0, sigma_E)
            dE = max(dE, 0)
        else:
            dE = dE_medio

        E = max(E - dE, 0)
        posiciones.append(x)
        energias_dep.append(dE)
        x += dx_cm
    return np.array(posiciones), np.array(energias_dep)

def simular_dosis(E0, N_protones, dx_cm, straggling=False):
    """ Acumula dosis D(z) de N protones """
    R = rango_CSDA(E0)
    x_bins = np.arange(0, R*1.3, dx_cm)
    dosis = np.zeros(len(x_bins)-1)
    rangos = []  # para medir straggling

    for _ in range(N_protones):
        pos, edep = simular_proton(E0, dx_cm, straggling)
        if len(pos) > 0:
            rangos.append(pos[-1])
        for xi, ei in zip(pos, edep):
            idx = int(xi/dx_cm)
            if 0 <= idx < len(dosis):
                dosis[idx] += ei
    x_centros = 0.5*(x_bins[:-1] + x_bins[1:])
    return x_centros, dosis, np.array(rangos)

# --- Simular SIN straggling ---
E0 = 150.0
dx = 0.01  # 0.1 mm = 0.01 cm
N = 10000

print("\nSimulando 10^4 protones SIN straggling...")
x_c, dosis_sin, rangos_sin = simular_dosis(E0, N, dx, straggling=False)
R_csda = rango_CSDA(E0)
pos_pico = x_c[np.argmax(dosis_sin)]

print(f"Pico de Bragg en z = {pos_pico:.2f} cm")
print(f"R_CSDA              = {R_csda:.2f} cm")

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x_c, dosis_sin/dosis_sin.max(), 'r-', lw=2, label='Dosis D(z)')
ax.fill_between(x_c, 0, dosis_sin/dosis_sin.max(), alpha=0.2, color='red')
ax.axvline(R_csda, color='blue', ls='--', label=f'$R_{{CSDA}}$={R_csda:.2f} cm')
ax.set_xlabel('Profundidad z (cm)')
ax.set_ylabel('Dosis relativa')
ax.set_title(f'Pico de Bragg — protón {E0:.0f} MeV (sin straggling)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p2_bragg_sin.png', dpi=120, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# INCISO (d): CON straggling + comparación
# ------------------------------------------------------------
print("\nSimulando 10^4 protones CON straggling...")
x_c2, dosis_con, rangos_con = simular_dosis(E0, N, dx, straggling=True)

# Ensanchamiento del pico
sigma_R = np.std(rangos_con)
print(f"Rango medio          = {np.mean(rangos_con):.2f} cm")
print(f"Ensanchamiento sigma_R = {sigma_R:.3f} cm = {sigma_R*10:.2f} mm")

# --- Comparación ---
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_c,  dosis_sin/dosis_sin.max(), 'b-', lw=2, label='Sin straggling')
ax.plot(x_c2, dosis_con/dosis_con.max(), 'r-', lw=2, label='Con straggling')
ax.axvline(R_csda, color='gray', ls='--', label=f'$R_{{CSDA}}$={R_csda:.2f} cm')
ax.set_xlabel('Profundidad z (cm)')
ax.set_ylabel('Dosis relativa')
ax.set_title(f'Pico de Bragg con/sin straggling — {E0:.0f} MeV\n$\\sigma_R$={sigma_R*10:.2f} mm')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(R_csda-3, R_csda+1)  # zoom en el pico
plt.tight_layout()
plt.savefig('p2_bragg_comparacion.png', dpi=120, bbox_inches='tight')
plt.show()

# --- Histograma de rangos (muestra la dispersión) ---
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(rangos_con, bins=50, density=True, color='tomato', alpha=0.7,
        label=f'Con straggling ($\\sigma_R$={sigma_R*10:.2f} mm)')
ax.axvline(R_csda, color='blue', ls='--', label='$R_{CSDA}$')
ax.set_xlabel('Rango de detención (cm)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución de rangos por straggling')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p2_histograma_rangos.png', dpi=120, bbox_inches='tight')
plt.show()
