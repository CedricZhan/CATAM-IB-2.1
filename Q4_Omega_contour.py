import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

mu = 0.5

def Omega(x, y, mu):
    r1 = np.sqrt((x + 1 - mu)**2 + y**2)
    r2 = np.sqrt((x - mu)**2 + y**2)

    return (
        -0.5 * mu * r1**2
        -0.5 * (1 - mu) * r2**2
        - mu / r1
        - (1 - mu) / r2
    )

def dOmega_dx(x, mu):
    r1 = np.abs(x + 1 - mu)
    r2 = np.abs(x - mu)

    return (-mu * (x + 1 - mu)
            - (1 - mu) * (x - mu)
            + mu * (x + 1 - mu) / r1**3
            + (1 - mu) * (x - mu) / r2**3)

L3 = fsolve(dOmega_dx, -1.2, args=(mu))[0]
L4 = fsolve(dOmega_dx, 0.0, args=(mu))[0]
L5 = fsolve(dOmega_dx, 1.2, args=(mu))[0]

L1 = (mu - 0.5,  np.sqrt(3)/2)
L2 = (mu - 0.5, -np.sqrt(3)/2)

xg = np.linspace(-3.0, 3.0, 800)
yg = np.linspace(-3.0, 3.0, 800)
X, Y = np.meshgrid(xg, yg)

Z = Omega(X, Y, mu)
Z[np.isinf(Z)] = np.nan

base_levels = np.linspace(-5, 1, 25)

omega_L3 = Omega(L3, 0, mu)
omega_L4 = Omega(L4, 0, mu)
omega_L5 = Omega(L5, 0, mu)

width = 0.18
n_local = 12

local_levels_L3 = np.linspace(omega_L3 - width, omega_L3 + width, n_local)
local_levels_L4 = np.linspace(omega_L4 - width, omega_L4 + width, n_local)
local_levels_L5 = np.linspace(omega_L5 - width, omega_L5 + width, n_local)

levels = np.unique(np.concatenate([
    base_levels,
    local_levels_L3,
    local_levels_L4,
    local_levels_L5
]))

plt.figure(figsize=(8, 8))

contours = plt.contour(X, Y, Z, levels=levels, linewidths=0.9)
plt.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

plt.scatter(-1 + mu, 0, c='red', s=90, label='P1', zorder=5)
plt.scatter(mu, 0, c='blue', s=90, label='P2', zorder=5)

plt.scatter(L3, 0, c='black', s=65, zorder=6)
plt.text(L3, 0.12, 'L3', ha='center', fontsize=11)

plt.scatter(L4, 0, c='black', s=65, zorder=6)
plt.text(L4, 0.12, 'L4', ha='center', fontsize=11)

plt.scatter(L5, 0, c='black', s=65, zorder=6)
plt.text(L5, 0.12, 'L5', ha='center', fontsize=11)

plt.scatter(*L1, c='green', s=85, zorder=6)
plt.text(L1[0], L1[1] + 0.12, 'L1', ha='center', fontsize=11)

plt.scatter(*L2, c='green', s=85, zorder=6)
plt.text(L2[0], L2[1] - 0.18, 'L2', ha='center', fontsize=11)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Contours of Omega with denser levels near L3, L4, L5')
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()