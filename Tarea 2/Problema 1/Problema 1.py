
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

#dado (gamma, k) calcular x(t)

def resolver_oscilador(gamma, k, t_eval, m=1.0):
    
    def edo(t, y):
        x, v = y
        return [v, -(gamma*v + k*x)/m]
    sol = solve_ivp(edo, [t_eval[0], t_eval[-1]], [1.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-8)
    return sol.y[0]


# Inciso (b):

def generar_dataset(N=3000, sigma=0.02, Nt=1000, t_max=10.0):
    t_eval = np.linspace(0, t_max, Nt)
    X = np.zeros((N, Nt))      # señales
    Theta = np.zeros((N, 2))   # parámetros (gamma, k)

    for i in range(N):
        gamma = np.random.uniform(0.05, 1.0)
        k     = np.random.uniform(1.0, 5.0)
        x_lim = resolver_oscilador(gamma, k, t_eval)
        ruido = np.random.normal(0, sigma, Nt)
        X[i] = x_lim + ruido
        Theta[i] = [gamma, k]
    return X, Theta, t_eval

print("Generando dataset...")
X, Theta, t_eval = generar_dataset(N=3000, sigma=0.02)
print("Dataset listo:", X.shape)

# Graficar algunas señales con distintos gamma y k 
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Variar gamma (k fijo)
for gamma in [0.1, 0.4, 0.9]:
    x_s = resolver_oscilador(gamma, 3.0, t_eval)
    axes[0].plot(t_eval, x_s, label=f'$\\gamma$={gamma}')
axes[0].set_title('Efecto del amortiguamiento $\\gamma$ (k=3)')
axes[0].set_xlabel('t'); axes[0].set_ylabel('x(t)')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Variar k (gamma fijo)
for k in [1.0, 3.0, 5.0]:
    x_s = resolver_oscilador(0.3, k, t_eval)
    axes[1].plot(t_eval, x_s, label=f'k={k}')
axes[1].set_title('Efecto de la rigidez k ($\\gamma$=0.3)')
axes[1].set_xlabel('t'); axes[1].set_ylabel('x(t)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p1_senales.png', dpi=120, bbox_inches='tight')
plt.show()

# Ejemplo de señal con ruido
fig, ax = plt.subplots(figsize=(8,4))
x_limpio = resolver_oscilador(0.3, 3.0, t_eval)
ax.plot(t_eval, X[0], lw=0.8, color='steelblue', label='Con ruido ($\\sigma$=0.02)')
ax.plot(t_eval, resolver_oscilador(*Theta[0], t_eval), 'r-', lw=1.5, label='Señal limpia')
ax.set_title(f'Señal con ruido: $\\gamma$={Theta[0,0]:.2f}, k={Theta[0,1]:.2f}')
ax.set_xlabel('t'); ax.set_ylabel('x(t)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p1_senal_ruido.png', dpi=120, bbox_inches='tight')
plt.show()


# Inciso (c):

X_train, X_test, y_train, y_test = train_test_split(
    X, Theta, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]} señales | Test: {X_test.shape[0]} señales")

#  Modelo 1: Random Forest 
print("\nEntrenando Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

#  Modelo 2: MLP (red neuronal) 
print("Entrenando MLP...")
mlp = MLPRegressor(hidden_layer_sizes=(128, 64),
                   activation='relu', max_iter=300,
                   random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

#  Calcular RMSE 
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2, axis=0))

rmse_rf  = rmse(y_test, pred_rf)
rmse_mlp = rmse(y_test, pred_mlp)

print("\n========== RESULTADOS ==========")
print(f"{'Modelo':<15}{'RMSE_gamma':>12}{'RMSE_k':>12}")
print(f"{'RandomForest':<15}{rmse_rf[0]:>12.4f}{rmse_rf[1]:>12.4f}")
print(f"{'MLP':<15}{rmse_mlp[0]:>12.4f}{rmse_mlp[1]:>12.4f}")

# Gráfico: predicho vs real 
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
modelos = [('Random Forest', pred_rf), ('MLP', pred_mlp)]
for fila, (nombre, pred) in enumerate(modelos):
    # gamma
    axes[fila,0].scatter(y_test[:,0], pred[:,0], s=8, alpha=0.4, color='steelblue')
    axes[fila,0].plot([0.05,1.0],[0.05,1.0],'r--')
    axes[fila,0].set_xlabel('$\\gamma$ real'); axes[fila,0].set_ylabel('$\\gamma$ predicho')
    axes[fila,0].set_title(f'{nombre}: $\\gamma$')
    axes[fila,0].grid(alpha=0.3)
    # k
    axes[fila,1].scatter(y_test[:,1], pred[:,1], s=8, alpha=0.4, color='darkorange')
    axes[fila,1].plot([1.0,5.0],[1.0,5.0],'r--')
    axes[fila,1].set_xlabel('k real'); axes[fila,1].set_ylabel('k predicho')
    axes[fila,1].set_title(f'{nombre}: k')
    axes[fila,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p1_pred_vs_real.png', dpi=120, bbox_inches='tight')
plt.show()


# Inciso (d)

sigmas = [0.0, 0.01, 0.02, 0.05, 0.10]
resultados_rmse = {'rf': [], 'mlp': []}
resultados_train = {'rf': [], 'mlp': []}  # error de entrenamiento (sobreajuste)

for sigma in sigmas:
    print(f"\n--- sigma = {sigma} ---")
    Xs, Ts, _ = generar_dataset(N=1500, sigma=sigma)  # N menor para rapidez
    Xtr, Xte, ytr, yte = train_test_split(Xs, Ts, test_size=0.2, random_state=0)

    # Random Forest
    rf_s = RandomForestRegressor(n_estimators=80, max_depth=20,
                                 random_state=0, n_jobs=-1)
    rf_s.fit(Xtr, ytr)
    resultados_rmse['rf'].append(rmse(yte, rf_s.predict(Xte)))
    resultados_train['rf'].append(rmse(ytr, rf_s.predict(Xtr)))

    # MLP
    mlp_s = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=250,
                         random_state=0, early_stopping=True)
    mlp_s.fit(Xtr, ytr)
    resultados_rmse['mlp'].append(rmse(yte, mlp_s.predict(Xte)))
    resultados_train['mlp'].append(rmse(ytr, mlp_s.predict(Xtr)))

resultados_rmse['rf']  = np.array(resultados_rmse['rf'])
resultados_rmse['mlp'] = np.array(resultados_rmse['mlp'])
resultados_train['rf'] = np.array(resultados_train['rf'])

#  Gráfico RMSE vs sigma 
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(sigmas, resultados_rmse['rf'][:,0], 'o-', label='RF', color='steelblue')
axes[0].plot(sigmas, resultados_rmse['mlp'][:,0], 's-', label='MLP', color='tomato')
axes[0].set_xlabel('$\\sigma$ (ruido)'); axes[0].set_ylabel('RMSE$_\\gamma$')
axes[0].set_title('Error en $\\gamma$ vs ruido'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(sigmas, resultados_rmse['rf'][:,1], 'o-', label='RF', color='steelblue')
axes[1].plot(sigmas, resultados_rmse['mlp'][:,1], 's-', label='MLP', color='tomato')
axes[1].set_xlabel('$\\sigma$ (ruido)'); axes[1].set_ylabel('RMSE$_k$')
axes[1].set_title('Error en k vs ruido'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p1_rmse_vs_sigma.png', dpi=120, bbox_inches='tight')
plt.show()

#  Sobreajuste: train vs test (Random Forest) 
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(sigmas, resultados_train['rf'][:,1], 'o--', label='Error TRAIN (k)', color='green')
ax.plot(sigmas, resultados_rmse['rf'][:,1], 'o-', label='Error TEST (k)', color='red')
ax.set_xlabel('$\\sigma$'); ax.set_ylabel('RMSE$_k$')
ax.set_title('Sobreajuste en Random Forest\n(brecha train-test)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p1_sobreajuste.png', dpi=120, bbox_inches='tight')
plt.show()

