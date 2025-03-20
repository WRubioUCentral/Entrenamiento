import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 1.0       # Longitud del dominio
N = 100       # Número de puntos en el espacio
dx = L / N    # Espaciado en x
dt = 0.0005    # Paso de tiempo
u = 2.0       # Velocidad de la onda
T = 1.0       # Tiempo total de simulación
steps = int(T / dt)  # Número de pasos de tiempo

# Discretización del dominio
x = np.linspace(0, L, N)
p = np.sin(2 * np.pi * x)  # Condición inicial
p_next = np.zeros_like(p)  # Estado futuro

# Simulación usando el método de diferencias finitas
for t in range(steps):
    p_next[1:] = p[1:] + u * dt / dx * (p[1:] - p[:-1])  # Esquema explícito upwind
    p = p_next.copy()

    if t % 50 == 0:  # Graficar cada ciertos pasos
        plt.plot(x, p, label=f"t={t*dt:.2f}")

# Configuración de la gráfica
plt.xlabel("x")
plt.ylabel("p(x,t)")
plt.title("Evolución de la ecuación dp/dt + u * dp/dx = 0")
plt.legend()
plt.grid()
plt.show()
