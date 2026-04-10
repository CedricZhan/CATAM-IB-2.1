import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

mu = 0.5

v0 = -1.50

def Omega(x, y, mu):
    r1 = np.sqrt((x + 1 - mu)**2 + y**2)
    r2 = np.sqrt((x - mu)**2 + y**2)

    return (
        -0.5 * mu * r1**2
        -0.5 * (1 - mu) * r2**2
        - mu / r1
        - (1 - mu) / r2
    )

def cr3bp(t, z, mu):
    x, y, u, v = z

    r1 = np.sqrt((x + 1 - mu)**2 + y**2)
    r2 = np.sqrt((x - mu)**2 + y**2)

    dOmega_dx = (
        -mu * (x + 1 - mu)
        - (1 - mu) * (x - mu)
        + mu * (x + 1 - mu) / r1**3
        + (1 - mu) * (x - mu) / r2**3
    )

    dOmega_dy = (
        -y
        + mu * y / r1**3
        + (1 - mu) * y / r2**3
    )

    dxdt = u
    dydt = v
    dudt = 2 * v - dOmega_dx
    dvdt = -2 * u - dOmega_dy

    return [dxdt, dydt, dudt, dvdt]

z0 = [0.2, 0.0, 0.0, v0]
t_span = (0.0, 15.0)
t_eval = np.linspace(0, 15, 5000)

sol = solve_ivp(cr3bp, t_span, z0, args=(mu,),
                t_eval=t_eval, rtol=1e-9, atol=1e-12)

x = sol.y[0]
y = sol.y[1]
u = sol.y[2]
v = sol.y[3]

J = 0.5 * v0**2 + Omega(0.2, 0.0, mu)

xg = np.linspace(-6, 6, 500)
yg = np.linspace(-6, 6, 500)
X, Y = np.meshgrid(xg, yg)

Z = Omega(X, Y, mu)

J_t = 0.5*(u**2 + v**2) + Omega(x, y, mu)
J0 = J_t[0]

print("max |J - J0| =", np.max(np.abs(J_t - J0)))

plt.figure(figsize=(7,7))

plt.contourf(X, Y, Z <= J, levels=[-1, 0, 1], alpha=0.3)

plt.contour(X, Y, Z, levels=[J], linewidths=2)

plt.plot(x, y, 'k', lw=1.5)

plt.scatter(-1 + mu, 0, c='red', s=80, label='P1')
plt.scatter(mu, 0, c='blue', s=80, label='P2')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'v0 = {v0}, J = {J:.3f}')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()